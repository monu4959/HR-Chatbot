# Labelling: taxonomy and annotation

This folder is the stage that produced the chatbot's dataset. It builds a taxonomy of recognition
topics from the raw messages with an LLM, then labels every message with one category and one
subcategory. Its output is `../data/annotated_results.csv` (byte-for-byte the file this pipeline wrote),
which the agent, the evals and the tests all use.

```
data/mockup_awards.csv  ──►  taxonomy (2 runs, cleaned)  ──►  data/cleaned_taxonomy_2.json  ──►  annotation  ──►  ../data/annotated_results.csv
1,000 messages                                                4 categories, 26 subcategories      Gemini 2.5 Flash     1,000 labelled rows
```

## What was run, in order

| Step | Script | What it did | Output |
|---|---|---|---|
| 1. Taxonomy, run 1 | `create_taxonomy_anthropic_2.py` | Batches of 20 messages, each batch extending the taxonomy (Gemini 2.5 Flash, despite the file name). Replies were capped at 2,000 tokens, so many were cut off mid-JSON. | per-batch files (not kept here) |
| 2. Recover | `recovery.py` | Extracted categories and subcategories from the cut-off replies. | `taxonomy_results/recovered_taxonomy.json` (36 categories, 85 subcategories) |
| 3. Compress | `compress.py` | Merged similar categories using hand-written merge groups, then added IDs (A, A1, …). | `taxonomy_results/compressed_taxonomy_with_ids.json` (6 / 47) |
| 4. Clean | by hand | Merged and renamed further. | `taxonomy_results/cleaned_taxonomy.json` (4 / 32) |
| 5. Taxonomy, run 2 | `taxonomyv2.py` | Same process with several API keys rotated on rate limits (free tier). | `taxonomy_results_1/compressed_taxonomy_with_ids.json` (8 / 65) |
| 6. Clean | by hand | Final merge to four categories. | **`data/cleaned_taxonomy_2.json`** (4 / 26), the taxonomy used for annotation |
| 7. Annotate | `run_topic_annotation_gemini.py` | Labelled all 1,000 messages in batches of 20 (Gemini 2.5 Flash, temperature 0.1). | `../data/annotated_results.csv` |

`create_taxonomy_FIXED.py` was written after run 1 to fix the cut-off replies (8,000 tokens, retries, sturdier JSON parsing). It re-ran only the first few batches, and its output did not feed the final taxonomy.

`annotationnoapi.py` is an alternative to step 7 that was tried but not used for the final data: it embeds
each message and each subcategory with sentence-transformers and picks the most similar subcategory,
with no API calls.

## The final taxonomy

| ID | Category | Subcategories | Records |
|---|---|---|---|
| A | Strategic Business Outcomes | 3 | 160 |
| B | Operational Excellence & Delivery | 9 | 298 |
| C | Creative & Brand Impact | 5 | 48 |
| D | Organizational & Team Enablement | 9 | 494 |

25 of the 26 subcategories were assigned; D5 (Diversity, Equity & Inclusion Support) never was.

## Re-running

The scripts read API keys from the environment, never from the code:

```bash
export GEMINI_API_KEYS="key1,key2"      # Google AI Studio keys, comma-separated (rotated on rate limits)
pip install google-generativeai json5 tqdm pandas numpy

cd labelling
python taxonomyv2.py                    # taxonomy (writes taxonomy_results_1/ and claude_responses_1/)
python run_topic_annotation_gemini.py   # annotation (writes output/annotated_results.csv)
```

Re-running gives a different taxonomy and labels: the steps call an LLM, and steps 4 and 6 were manual.
The committed files are the ones the chatbot was built on.

## Known limitations

- **The labels were not checked by a person.** Every category count the chatbot reports assumes Gemini's
  labels are right. Agreement could be measured the same way as the grader in `evals/grader_validation/`.
- **The cleaning steps (4 and 6) were done by hand** and are not reproducible from code.
- **The categories are uneven.** D holds 49% of records, and D1 (Team Collaboration & Integration) alone
  holds 24%, so it may act as a catch-all.
- The input is `mockup_awards.csv`: message, award title, recipient title and nominator title only, with
  no names, departments or dates. That is why the chatbot refuses date questions.
