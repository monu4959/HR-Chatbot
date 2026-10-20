#!/usr/bin/env python3
"""
Local Topic Annotation using Sentence Transformers
----------------------------------------------------
Annotates award messages with taxonomy categories and subcategories
using semantic similarity — no API, no internet after model download.

How it works:
  1. Embeds every subcategory (name + category context) into a vector
  2. Embeds each award message into a vector
  3. Finds the closest subcategory via cosine similarity
  4. Assigns that category + subcategory to the record

Install once:
    pip install sentence-transformers pandas numpy tqdm

Model downloaded once (~90MB), cached locally after that.

Usage:
    python annotate_local.py
"""

import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: Please install sentence-transformers first:")
    print("  pip install sentence-transformers")
    sys.exit(1)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME   = "all-MiniLM-L6-v2"   # ~90MB, fast on CPU, good accuracy
BATCH_SIZE   = 64                    # messages encoded per batch (tune for RAM)
TOP_N        = 3                     # how many top matches to show per record


# =============================================================================
# LOAD TAXONOMY
# =============================================================================

def load_taxonomy(path: str) -> list[dict]:
    """
    Flatten taxonomy into a list of subcategory dicts, each with:
      category_id, category_name, subcategory_id, subcategory_name, embed_text
    embed_text is what we actually embed — enriched with context for better matching.
    """
    with open(path) as f:
        taxonomy = json.load(f)

    entries = []
    for cat_name, cat_data in taxonomy["recognition_taxonomy"].items():
        cat_id   = cat_data["id"]
        cat_desc = cat_data.get("description", "")

        for sub in cat_data["subcategories"]:
            sub_id   = sub["id"]
            sub_name = sub["name"]

            # Enrich the embed text with category context so similarity
            # is more meaningful than just matching the short subcategory name
            embed_text = (
                f"{sub_name}. "
                f"Category: {cat_name}. "
                f"{cat_desc}"
            )

            entries.append({
                "category_id":      cat_id,
                "category_name":    cat_name,
                "subcategory_id":   sub_id,
                "subcategory_name": sub_name,
                "embed_text":       embed_text,
            })

    return entries


# =============================================================================
# FIND FILE HELPERS
# =============================================================================

def find_file(candidates: list[str], label: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    custom = input(f"Path to {label}: ").strip()
    if not os.path.exists(custom):
        print(f"ERROR: Not found: {custom}")
        sys.exit(1)
    return custom


def find_taxonomy() -> str:
    return find_file([
        "taxonomy_results/cleaned_taxonomy.json",
        "cleaned_taxonomy.json",
        "taxonomy_results/compressed_taxonomy_with_ids.json",
        "compressed_taxonomy_with_ids.json",
        "data/compressed_taxonomy_with_ids.json",
    ], "taxonomy JSON file")


def find_csv() -> str:
    return find_file([
        "mockup_awards.csv",
        "data/mockup_awards.csv",
    ], "awards CSV file")


# =============================================================================
# FORMAT MESSAGE
# =============================================================================

def build_message_text(row: pd.Series) -> str:
    """
    Combine available fields into a single string to embed.
    Handles both original wide schema and mockup CSV schema.
    """
    parts = []

    def grab(*keys):
        for k in keys:
            v = row.get(k)
            if v is not None and pd.notna(v) and str(v).strip():
                return str(v).strip()
        return None

    award_title = grab("title", "award_title")
    award_reason = grab("award_reason")
    message = grab("message")
    rec_title = grab("rec_job_title", "recipient_title")
    nom_title = grab("nom_job_title", "nominator_title")

    if award_title:   parts.append(f"Award: {award_title}")
    if award_reason:  parts.append(f"Reason: {award_reason}")
    if message:       parts.append(f"Message: {message}")
    if rec_title:     parts.append(f"Recipient: {rec_title}")
    if nom_title:     parts.append(f"Nominator: {nom_title}")

    return " | ".join(parts) if parts else str(row.get("message", ""))


# =============================================================================
# ANNOTATE
# =============================================================================

def annotate(
    df: pd.DataFrame,
    taxonomy_entries: list[dict],
    model: SentenceTransformer,
) -> pd.DataFrame:
    """
    Embed all messages and subcategories, then assign best-match subcategory
    to each message using cosine similarity.
    """

    # ── Embed subcategories ──────────────────────────────────────────────────
    print("\nEmbedding taxonomy subcategories...")
    sub_texts = [e["embed_text"] for e in taxonomy_entries]
    sub_embeddings = model.encode(
        sub_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalised = cosine sim is just dot product
    )

    # ── Build message texts ──────────────────────────────────────────────────
    print("\nBuilding message texts...")
    df = df.copy()
    df["_message_text"] = df.apply(build_message_text, axis=1)

    # ── Embed messages in batches ────────────────────────────────────────────
    print(f"\nEmbedding {len(df)} messages (batch size {BATCH_SIZE})...")
    msg_embeddings = model.encode(
        df["_message_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # ── Compute similarity & assign labels ───────────────────────────────────
    print("\nAssigning categories via cosine similarity...")

    # similarity matrix: (num_messages, num_subcategories)
    sim_matrix = np.dot(msg_embeddings, sub_embeddings.T)

    results = []
    for i, scores in enumerate(tqdm(sim_matrix, desc="Annotating")):
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_entry = taxonomy_entries[best_idx]

        # Top-N matches for transparency
        top_indices = np.argsort(scores)[::-1][:TOP_N]
        top_matches = [
            f"{taxonomy_entries[j]['subcategory_name']} "
            f"({taxonomy_entries[j]['category_name']}, "
            f"score={scores[j]:.3f})"
            for j in top_indices
        ]

        results.append({
            "category_id":      best_entry["category_id"],
            "category_name":    best_entry["category_name"],
            "subcategory_id":   best_entry["subcategory_id"],
            "subcategory_name": best_entry["subcategory_name"],
            "similarity_score": round(best_score, 4),
            "top_matches":      " | ".join(top_matches),
        })

    result_df = pd.DataFrame(results)

    # Drop the temp column and merge results
    df = df.drop(columns=["_message_text"])
    df = df.reset_index(drop=True)
    annotated = pd.concat([df, result_df], axis=1)

    return annotated


# =============================================================================
# REPORT
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"  Total records annotated : {len(df)}")
    print(f"  Avg similarity score    : {df['similarity_score'].mean():.3f}")
    print(f"  Min similarity score    : {df['similarity_score'].min():.3f}")
    print(f"  Max similarity score    : {df['similarity_score'].max():.3f}")

    print("\n  Category Distribution:")
    cat_counts = (
        df.groupby(["category_id", "category_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    for _, row in cat_counts.iterrows():
        pct = row["count"] / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {row['category_id']}  {row['category_name']:<35} "
              f"{row['count']:>4}  ({pct:5.1f}%)  {bar}")

    print("\n  Top 10 Subcategories:")
    sub_counts = (
        df.groupby(["subcategory_id", "subcategory_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    for _, row in sub_counts.iterrows():
        pct = row["count"] / len(df) * 100
        print(f"    {row['subcategory_id']:<4}  {row['subcategory_name']:<40}  "
              f"{row['count']:>4}  ({pct:5.1f}%)")

    # Low-confidence records
    low_conf = df[df["similarity_score"] < 0.3]
    if len(low_conf) > 0:
        print(f"\n  Low confidence (score < 0.30) : {len(low_conf)} records")
        print("  Consider reviewing these — they may not fit any category well.")

    print("=" * 60)


# =============================================================================
# INPUT HELPER
# =============================================================================

def ask(prompt: str, default: str, validate=None) -> str:
    while True:
        val = input(f"{prompt} [{default}]: ").strip() or default
        if validate is None or validate(val):
            return val
        print("  Invalid — please try again.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Local Topic Annotation — Sentence Transformers (No API)")
    print("=" * 60)

    # ── Find files ───────────────────────────────────────────────────────────
    taxonomy_path = find_taxonomy()
    csv_path      = find_csv()

    print(f"\n  Taxonomy : {taxonomy_path}")
    print(f"  CSV      : {csv_path}")

    # ── Sample size ──────────────────────────────────────────────────────────
    sample_size = int(ask(
        "\nHow many records to annotate (0 = all)",
        "0",
        lambda x: x.isdigit(),
    ))

    output_file = ask(
        "Output CSV filename",
        "output/annotated_results.csv",
    )

    if input("\nProceed? [Y/n]: ").strip().lower() == "n":
        sys.exit(0)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # ── Load taxonomy ─────────────────────────────────────────────────────────
    print(f"\nLoading taxonomy from {taxonomy_path}...")
    taxonomy_entries = load_taxonomy(taxonomy_path)
    print(f"  {len(taxonomy_entries)} subcategories loaded across "
          f"{len(set(e['category_name'] for e in taxonomy_entries))} categories")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} records, columns: {list(df.columns)}")

    if "message" not in df.columns:
        print("ERROR: CSV must have a 'message' column.")
        sys.exit(1)

    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    if sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df)} records")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading sentence transformer model: {MODEL_NAME}")
    print("  (Downloads ~90MB on first run, cached after that)")
    t0    = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Annotate ──────────────────────────────────────────────────────────────
    t0       = time.time()
    annotated = annotate(df, taxonomy_entries, model)
    elapsed  = time.time() - t0
    print(f"\n  Annotation complete in {elapsed:.1f}s "
          f"({len(df)/elapsed:.0f} records/sec)")

    # ── Save ──────────────────────────────────────────────────────────────────
    annotated.to_csv(output_file, index=False)
    print(f"\n  Saved -> {output_file}")

    # ── Report ────────────────────────────────────────────────────────────────
    print_summary(annotated)

    # ── Sample preview ────────────────────────────────────────────────────────
    show = input("\nShow 5 sample annotations? [y/N]: ").strip().lower()
    if show == "y":
        sample = annotated.sample(n=min(5, len(annotated)), random_state=1)
        for _, row in sample.iterrows():
            print(f"\n  {'─'*55}")
            msg = str(row.get("message", ""))[:120]
            print(f"  Message    : {msg}...")
            print(f"  Category   : {row['category_id']}  {row['category_name']}")
            print(f"  Subcategory: {row['subcategory_id']}  {row['subcategory_name']}")
            print(f"  Confidence : {row['similarity_score']:.3f}")
            print(f"  Top matches: {row['top_matches']}")


if __name__ == "__main__":
    main()