"""
agent/cache.py

Two caches, both SQLite files under .cache/ (git-ignored).

  LLM cache     Every model call, keyed on the exact prompt + model + parameters (installed as
                LangChain's global cache, so every tier and every tool is covered). Same input →
                same output, so it never returns a wrong answer — but a hit is not *fresh*: it
                won't show provider-side drift or run-to-run variation. Evals can turn it off.
  Answer cache  Whole answers to standalone questions (no conversation history), keyed on the
                normalized question + data hash + code hash + provider/models. A hit skips the
                whole pipeline. Follow-ups, grader-refused answers and grader errors are never
                stored, and any change to the CSV or to agent/ or tools/ code starts a new cache.

Env:
  LLM_CACHE=on|off          default on
  ANSWER_CACHE=on|off       default on
  ANSWER_CACHE_TTL_HOURS    default 168 (a week)
  CACHE_DIR                 default ./.cache

CLI:
  python -m agent.cache stats
  python -m agent.cache clear [llm|answers|all]
"""

import os
import re
import sys
import json
import time
import sqlite3
import hashlib
import warnings
import threading
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager

from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.load import dumps, loads

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / ".cache"))
LLM_DB    = CACHE_DIR / "llm_cache.sqlite"
ANSWER_DB = CACHE_DIR / "answer_cache.sqlite"

_stats = {"llm_hits": 0, "llm_misses": 0, "answer_hits": 0, "answer_misses": 0}
_lock  = threading.Lock()


def _count(key: str):
    with _lock:
        _stats[key] += 1


def _enabled(var: str) -> bool:
    return os.getenv(var, "on").strip().lower() not in ("off", "0", "false", "no")


@contextmanager
def _db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── LLM cache ─────────────────────────────────────────────────────────────────

# Message fields that are bookkeeping, not model input. They change between runs (LangChain adds
# "total_cost": 0 to a cached reply's usage, providers return timings), and an earlier reply is part of
# the next call's prompt — so leaving them in the key makes one hit cause a miss on the next call.
_VOLATILE_MESSAGE_FIELDS = {"id", "response_metadata", "usage_metadata"}


def _strip_volatile(node):
    if isinstance(node, list):
        return [_strip_volatile(n) for n in node]
    if isinstance(node, dict):
        if node.get("lc") == 1 and isinstance(node.get("kwargs"), dict):  # a serialized LangChain message
            kwargs = {k: _strip_volatile(v) for k, v in node["kwargs"].items() if k not in _VOLATILE_MESSAGE_FIELDS}
            return {**node, "kwargs": kwargs}
        return {k: _strip_volatile(v) for k, v in node.items()}
    return node


def stable_prompt(prompt: str) -> str:
    """The serialized messages with bookkeeping fields removed, in a canonical key order."""
    try:
        return json.dumps(_strip_volatile(json.loads(prompt)), sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return prompt  # not a serialized message list (plain-text LLM prompt) — use as is


class SQLiteLLMCache(BaseCache):
    def __init__(self, path: Path | None = None):
        self.path = Path(path or LLM_DB)
        with _db(self.path) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS llm_cache "
                      "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)")

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\x00{stable_prompt(prompt)}".encode("utf-8")).hexdigest()

    def lookup(self, prompt: str, llm_string: str):
        with _db(self.path) as c:
            row = c.execute("SELECT value FROM llm_cache WHERE key = ?", (self._key(prompt, llm_string),)).fetchone()
        if row is not None:
            try:
                with warnings.catch_warnings():  # langchain_core.load.loads is marked beta
                    warnings.simplefilter("ignore")
                    generations = loads(row[0])
                _count("llm_hits")
                return generations
            except Exception:
                pass  # unreadable entry (e.g. written by another LangChain version) — treat as a miss
        _count("llm_misses")
        return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        with _db(self.path) as c:
            c.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
                      (self._key(prompt, llm_string), dumps(return_val), time.time()))

    def clear(self, **kwargs) -> None:
        with _db(self.path) as c:
            c.execute("DELETE FROM llm_cache")


def ensure_llm_cache():
    """Install (or remove) the global LLM cache to match LLM_CACHE. Cheap to call on every get_llm()."""
    current = get_llm_cache()
    if _enabled("LLM_CACHE"):
        if not isinstance(current, SQLiteLLMCache):
            set_llm_cache(SQLiteLLMCache())
    elif current is not None:
        set_llm_cache(None)


# ── Answer cache ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def code_version() -> str:
    """Hash of everything that shapes an answer — editing any agent/ or tools/ file starts a fresh cache."""
    h = hashlib.sha256()
    for path in sorted([*ROOT.glob("agent/*.py"), *ROOT.glob("tools/*.py")]):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def data_version() -> str:
    data_path = Path(os.getenv("DATA_PATH", "./data/annotated_results.csv"))
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    return hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]


def normalize_question(question: str) -> str:
    """Exact-match normalization only: case, whitespace and trailing punctuation.
    Deliberately not semantic — 'how many in Creative…' and 'how many in Strategic…' embed almost
    identically but have different answers."""
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip(" ?.!")


class AnswerCache:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or ANSWER_DB)
        with _db(self.path) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS answers "
                      "(key TEXT PRIMARY KEY, question TEXT, value TEXT NOT NULL, created_at REAL NOT NULL)")

    @staticmethod
    def _key(question: str, models: dict) -> str:
        parts = [normalize_question(question), data_version(), code_version(), models]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, question: str, models: dict) -> dict | None:
        ttl = float(os.getenv("ANSWER_CACHE_TTL_HOURS", "168")) * 3600
        with _db(self.path) as c:
            row = c.execute("SELECT value, created_at FROM answers WHERE key = ?",
                            (self._key(question, models),)).fetchone()
        if row is None or time.time() - row[1] > ttl:
            _count("answer_misses")
            return None
        _count("answer_hits")
        return json.loads(row[0])

    def put(self, question: str, models: dict, result: dict) -> None:
        with _db(self.path) as c:
            c.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
                      (self._key(question, models), question, json.dumps(result, ensure_ascii=False), time.time()))

    def clear(self) -> None:
        with _db(self.path) as c:
            c.execute("DELETE FROM answers")


def get_answer_cache() -> AnswerCache | None:
    """The answer cache, or None when ANSWER_CACHE=off (read on every call, so evals can switch it off)."""
    return AnswerCache() if _enabled("ANSWER_CACHE") else None


def is_cacheable(result: dict) -> bool:
    """Only cache answers the pipeline is confident in."""
    return (not result.get("was_refused") and not result.get("grader_error")
            and result.get("route_used") is not None
            and result.get("answer") != "Sorry, I could not generate a response.")


# ── Stats / CLI ───────────────────────────────────────────────────────────────

def _rows(path: Path, table: str) -> int:
    if not path.exists():
        return 0
    with _db(path) as c:
        try:
            return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def cache_stats() -> dict:
    with _lock:
        counters = dict(_stats)
    return {
        **counters,
        "llm_cache_enabled":    _enabled("LLM_CACHE"),
        "answer_cache_enabled": _enabled("ANSWER_CACHE"),
        "llm_entries":          _rows(LLM_DB, "llm_cache"),
        "answer_entries":       _rows(ANSWER_DB, "answers"),
        "code_version":         code_version(),
        "data_version":         data_version(),
    }


def clear(which: str = "all"):
    if which in ("llm", "all") and LLM_DB.exists():
        SQLiteLLMCache().clear()
    if which in ("answers", "all") and ANSWER_DB.exists():
        AnswerCache().clear()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "stats":
        print(json.dumps(cache_stats(), indent=2))
    elif cmd == "clear":
        which = sys.argv[2] if len(sys.argv) > 2 else "all"
        clear(which)
        print(f"Cleared: {which}")
    else:
        sys.exit("usage: python -m agent.cache [stats | clear [llm|answers|all]]")
