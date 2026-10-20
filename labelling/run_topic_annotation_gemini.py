#!/usr/bin/env python3
"""
Topic Annotation Pipeline - Gemini 2.5 Flash
Annotates employee recognition records with taxonomy categories.

Configuration:
- Model: gemini-2.5-flash
- Batch size: 20 records per batch
- API keys: 5 keys with rotation (retry same batch 2x per key, then rotate)
- Failed records saved to failed_records.csv for manual retry
"""

import json
import time
import pandas as pd
import numpy as np
import urllib.request
import urllib.error
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — update these before running
# ─────────────────────────────────────────────

HERE            = Path(__file__).resolve().parent
AWARD_CSV       = HERE / "data" / "mockup_awards.csv"
TAXONOMY_JSON   = HERE / "data" / "cleaned_taxonomy_2.json"
OUTPUT_DIR      = HERE / "output"

OUTPUT_FILE     = "annotated_results.csv"
FAILED_FILE     = "failed_records.csv"

# Keys come from the environment, never the code: GEMINI_API_KEYS="key1,key2,..." (rotated on 429s)
API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]

BATCH_SIZE      = 20
RETRIES_PER_KEY = 2          # retry same key this many times before rotating
SLEEP_BETWEEN   = 4          # seconds to wait between batches (free tier: 15 RPM)

MODEL           = "gemini-2.5-flash"
GEMINI_URL      = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)

# ─────────────────────────────────────────────
# LOAD FILES
# ─────────────────────────────────────────────

def load_data():
    df = pd.read_csv(AWARD_CSV)
    # Keep only needed columns
    cols = [c for c in ["message", "award_title", "recipient_title", "nominator_title"] if c in df.columns]
    df = df[cols].reset_index(drop=True)
    print(f"Loaded {len(df)} records from CSV")

    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    print(f"Loaded taxonomy with {len(taxonomy['recognition_taxonomy'])} top-level categories")
    return df, taxonomy


def build_taxonomy_text(taxonomy):
    """Convert taxonomy JSON into a compact readable string for the prompt."""
    lines = []
    for cat_name, cat_data in taxonomy["recognition_taxonomy"].items():
        lines.append(f"[{cat_data['id']}] {cat_name}")
        for sub in cat_data["subcategories"]:
            lines.append(f"    [{sub['id']}] {sub['name']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an HR researcher annotating employee recognition messages.
You will be given a batch of recognition messages and a taxonomy of categories.
For each message, assign the single best matching category and subcategory from the taxonomy.

Return ONLY a valid JSON array — no explanation, no markdown, no code fences.
Each element must have exactly these fields:
  "index"          : integer (the index provided with the message)
  "category_id"    : string (e.g. "A")
  "category_name"  : string (e.g. "Strategic Business Outcomes")
  "subcategory_id" : string (e.g. "A1")
  "subcategory_name": string (e.g. "Business Growth & Client Success")

Example output format:
[
  {"index": 0, "category_id": "A", "category_name": "Strategic Business Outcomes", "subcategory_id": "A1", "subcategory_name": "Business Growth & Client Success"},
  {"index": 1, "category_id": "D", "category_name": "Organizational & Team Enablement", "subcategory_id": "D1", "subcategory_name": "Team Collaboration & Integration"}
]"""


def build_prompt(batch_df, taxonomy_text):
    messages_block = ""
    for i, row in batch_df.iterrows():
        local_idx = batch_df.index.get_loc(i)
        messages_block += f"\n[{local_idx}] {row['message']}\n"

    return f"""TAXONOMY:
{taxonomy_text}

RECOGNITION MESSAGES TO ANNOTATE:
{messages_block}

Return a JSON array with one object per message using the exact format specified."""


# ─────────────────────────────────────────────
# GEMINI API CALL
# ─────────────────────────────────────────────

def call_gemini(prompt, api_key):
    """Call Gemini API using only stdlib urllib (no google-generativeai needed)."""
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
            "thinkingConfig": {
                "thinkingBudget": 0      # 0 = disable thinking entirely
            }
        }
    }   

    data = json.dumps(payload).encode("utf-8")
    url  = f"{GEMINI_URL}?key={api_key}"
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    return text


# ─────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────

def parse_response(text, expected_count):
    """
    Extract JSON array from model response.
    Returns list of dicts or raises ValueError if parsing fails / count mismatch.
    """
    # Strip markdown fences if model added them despite instructions
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    clean = clean.strip()

    parsed = json.loads(clean)

    if not isinstance(parsed, list):
        raise ValueError("Response is not a JSON array")

    if len(parsed) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(parsed)}")

    # Validate required fields
    required = {"index", "category_id", "category_name", "subcategory_id", "subcategory_name"}
    for item in parsed:
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Missing fields {missing} in item {item}")

    return parsed


# ─────────────────────────────────────────────
# BATCH PROCESSOR WITH KEY ROTATION
# ─────────────────────────────────────────────

def process_batch(batch_df, prompt, batch_num):
    """
    Try to annotate a batch.
    Strategy: retry same key RETRIES_PER_KEY times, then rotate to next key.
    If all 5 keys exhausted → return None (batch goes to failed).
    """
    for key_idx, api_key in enumerate(API_KEYS):
        for attempt in range(1, RETRIES_PER_KEY + 1):
            try:
                print(f"  Batch {batch_num:02d} | Key {key_idx+1} | Attempt {attempt} ...", end=" ")
                raw = call_gemini(prompt, api_key)
                parsed = parse_response(raw, len(batch_df))
                print(f"✅ {len(parsed)} records annotated")
                return parsed

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                print(f"❌ HTTP {e.code}: {body[:120]}")
                if e.code in (429, 500, 503):
                    # Rate limit or server error — wait before retrying
                    wait = 10 if attempt == 1 else 20
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    # 400 bad request etc — rotating key won't help, but try anyway
                    break

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"❌ Parse error: {e}")
                # Parse errors are response quality issues — try next attempt

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                time.sleep(5)

        print(f"  Key {key_idx+1} exhausted, rotating to next key...")

    print(f"  ⚠️  All keys failed for batch {batch_num} — sending to failed_records.csv")
    return None


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    if not API_KEYS:
        raise SystemExit("Set GEMINI_API_KEYS (comma-separated Google AI Studio keys) before running.")
    # ── Setup ──────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    failed_path = Path(OUTPUT_DIR) / FAILED_FILE

    # ── Load data ──────────────────────────────
    df, taxonomy = load_data()
    taxonomy_text = build_taxonomy_text(taxonomy)

    # ── Assign batch IDs ───────────────────────
    df["batch_id"] = np.ceil((np.arange(len(df)) + 1) / BATCH_SIZE).astype(int)
    total_batches  = df["batch_id"].max()
    print(f"\nTotal batches: {total_batches} ({BATCH_SIZE} records each)\n")

    # ── Process batches ────────────────────────
    all_results  = []   # list of result dicts
    failed_rows  = []   # list of original df rows that failed

    for batch_num in range(1, total_batches + 1):
        batch_df    = df[df["batch_id"] == batch_num].copy()
        batch_df    = batch_df.reset_index()   # keeps original index in 'index' col
        orig_indices = batch_df["index"].tolist()
        batch_df    = batch_df.drop(columns=["batch_id"])

        prompt  = build_prompt(batch_df, taxonomy_text)
        results = process_batch(batch_df, prompt, batch_num)

        if results is None:
            # Entire batch failed
            for orig_idx in orig_indices:
                failed_rows.append(df.loc[orig_idx, ["message", "award_title", "recipient_title", "nominator_title"]].to_dict())
        else:
            for item in results:
                local_idx = item["index"]
                orig_idx  = orig_indices[local_idx]
                row       = df.loc[orig_idx].to_dict()
                row.update({
                    "category_id"    : item["category_id"],
                    "category_name"  : item["category_name"],
                    "subcategory_id" : item["subcategory_id"],
                    "subcategory_name": item["subcategory_name"],
                })
                all_results.append(row)

        # Respect free tier rate limit
        if batch_num < total_batches:
            time.sleep(SLEEP_BETWEEN)

    # ── Save results ───────────────────────────
    result_cols = ["message", "award_title", "recipient_title", "nominator_title",
                   "category_id", "category_name", "subcategory_id", "subcategory_name"]

    if all_results:
        results_df = pd.DataFrame(all_results)
        # Keep only expected columns in order (drop batch_id etc)
        results_df = results_df[[c for c in result_cols if c in results_df.columns]]
        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Annotated results saved → {output_path}  ({len(results_df)} records)")
    else:
        print("\n⚠️  No records were successfully annotated.")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df = failed_df[[c for c in ["message", "award_title", "recipient_title", "nominator_title"] if c in failed_df.columns]]
        failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"⚠️  Failed records saved  → {failed_path}  ({len(failed_df)} records)")
    else:
        print("✅ No failed records — all batches processed successfully!")

    # ── Summary ────────────────────────────────
    total      = len(df)
    successful = len(all_results)
    failed     = len(failed_rows)
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total records   : {total}")
    print(f"Annotated       : {successful} ({successful/total*100:.1f}%)")
    print(f"Failed          : {failed}     ({failed/total*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
