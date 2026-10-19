import os
import tempfile

# Tests must not read or write the real caches in .cache/ — tests that need a cache build their own.
os.environ["LLM_CACHE"] = "off"
os.environ["ANSWER_CACHE"] = "off"
os.environ["CACHE_DIR"] = tempfile.mkdtemp(prefix="test-cache-")
os.environ.setdefault("LANGSMITH_TRACING", "false")
