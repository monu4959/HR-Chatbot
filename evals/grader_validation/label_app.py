"""
evals/grader_validation/label_app.py

Labelling UI for grader validation, in two modes:

  blind     Items listed in blind_items.json. No suggestion is shown — these measure how well the
            Claude-drafted labels agree with a human, so they must stay uninfluenced. Shown first.
  reviewed  Every other item. Claude's draft label (drafts.jsonl) is shown with its reason and
            pre-selected; you confirm or correct it. Whether you changed it is recorded.

You never see which model wrote an answer, whether it was deliberately broken, or what the
automatic checks said.

  streamlit run evals/grader_validation/label_app.py --server.address localhost

Labels are saved to evals/grader_validation/labels.jsonl after every item.
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

HERE        = Path(__file__).resolve().parent
ITEMS_PATH  = HERE / "items.jsonl"
LABELS_PATH = HERE / "labels.jsonl"
DRAFTS_PATH = HERE / "drafts.jsonl"
BLIND_PATH  = HERE / "blind_items.json"
CHOICES     = ["yes", "no", "unsure"]

GUIDELINES = """
**Faithful** — is *every* factual claim in the answer supported by the tool output?
- Claims = numbers, counts, percentages, names, job titles, categories, rankings, quotes, statements about what the data shows.
- Supported = stated in the tool output, or follows by simple arithmetic / reordering.
- One unsupported claim → **no**. Ignore formatting, the "Route:" line, offers to help, generic caveats.
- Saying the info isn't available is faithful when the tool output doesn't contain it.

**Correct** — does the answer correctly answer the question, judged against the reference?
- Exact-value references: the answer must state those values (formatting/rounding within tolerance is fine).
- "Cannot be answered" references: the answer must decline or say the info isn't available — inventing data → **no**.
- Open-ended references: **yes** if it addresses the question directly and stays on topic.
- Judge correctness only, not style. A faithful answer can still be incorrect (the tool itself was wrong), and vice versa.

Use **unsure** sparingly — those items are left out of the agreement numbers.
"""


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_labels() -> dict:
    return {rec["item_id"]: rec for rec in _jsonl(LABELS_PATH)}


def save_labels(labels: dict):
    LABELS_PATH.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in labels.values()),
                           encoding="utf-8")


st.set_page_config(page_title="Grader labelling", layout="wide")
items  = _jsonl(ITEMS_PATH)
labels = load_labels()
drafts = {d["item_id"]: d for d in _jsonl(DRAFTS_PATH)}
blind  = set(json.loads(BLIND_PATH.read_text(encoding="utf-8"))["item_ids"]) if BLIND_PATH.exists() else set()

# Blind items first, then the ones with suggestions
order = [it for it in items if it["item_id"] in blind] + [it for it in items if it["item_id"] not in blind]


def next_unlabelled() -> int | None:
    for i, it in enumerate(order):
        if it["item_id"] not in labels and (it["item_id"] in blind or it["item_id"] in drafts or not blind):
            return i
    return None


if "idx" not in st.session_state:
    st.session_state.idx = next_unlabelled() or 0

with st.sidebar:
    n_blind  = sum(i in labels for i in blind)
    n_review = sum(it["item_id"] in labels for it in order if it["item_id"] not in blind)
    st.progress(len(labels) / len(items), text=f"{len(labels)} / {len(items)} labelled")
    if blind:
        st.caption(f"🔒 Blind: {n_blind} / {len(blind)}   ·   ✏️ Review: {n_review} / {len(items) - len(blind)}")
    with st.expander("Labelling guidelines", expanded=not labels):
        st.markdown(GUIDELINES)
    jump = st.number_input("Go to item #", 1, len(order), st.session_state.idx + 1)
    if jump - 1 != st.session_state.idx and st.button("Go"):
        st.session_state.idx = jump - 1
        st.rerun()
    if st.button("Next unlabelled"):
        nxt = next_unlabelled()
        if nxt is not None:
            st.session_state.idx = nxt
            st.rerun()

item     = order[st.session_state.idx]
is_blind = item["item_id"] in blind
draft    = None if is_blind else drafts.get(item["item_id"])
prior    = labels.get(item["item_id"], {})

st.markdown(f"### Item {st.session_state.idx + 1} of {len(order)}" + ("  ✅" if prior else ""))
if is_blind:
    st.caption("🔒 Blind item — label it without suggestions.")
elif blind and not draft:
    st.warning("Suggestions for this item aren't ready yet. Label the blind items first, or label this one yourself.")

if item.get("history"):
    st.caption("Earlier in the conversation: " + " → ".join(item["history"]))
st.markdown(f"**Question:** {item['question']}")
st.info(f"**Reference (computed from the data):** {item['reference']}")

left, right = st.columns(2)
with left:
    st.markdown("**Tool output** (what the assistant had to work with)")
    if item.get("tool_output_truncated"):
        st.warning("This tool output was cut off at 4,000 characters when it was saved.")
    st.code(item["tool_output"] or "(no tool output)", language=None, wrap_lines=True)
with right:
    st.markdown("**Answer**")
    with st.container(border=True):
        st.markdown(item["answer"])

if draft:
    st.success(f"**Suggested label (Claude):** Faithful: **{draft['faithful']}** · Correct: **{draft['correct']}**  \n"
               f"_{draft.get('reason', '')}_  \nCheck it against the item — change either answer if you disagree.")


def _default(field: str):
    if prior:
        return CHOICES.index(prior[field])
    if draft and draft.get(field) in CHOICES:
        return CHOICES.index(draft[field])
    return None


with st.form(key=f"form_{item['item_id']}"):
    c1, c2 = st.columns(2)
    faithful = c1.radio("Faithful to the tool output?", CHOICES, horizontal=True, index=_default("faithful"))
    correct  = c2.radio("Correct answer to the question?", CHOICES, horizontal=True, index=_default("correct"))
    note = st.text_input("Note (optional — e.g. which claim is unsupported)", value=prior.get("note", ""))
    if st.form_submit_button("Confirm & next" if draft else "Save & next", type="primary"):
        if faithful is None or correct is None:
            st.error("Pick an answer for both questions.")
        else:
            rec = {
                "item_id": item["item_id"], "faithful": faithful, "correct": correct, "note": note,
                "mode": "blind" if (is_blind or not draft) else "reviewed",
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
            }
            if draft:
                rec.update(draft_faithful=draft["faithful"], draft_correct=draft["correct"],
                           changed=(faithful, correct) != (draft["faithful"], draft["correct"]))
            labels[item["item_id"]] = rec
            save_labels(labels)
            nxt = next_unlabelled()
            st.session_state.idx = nxt if nxt is not None else st.session_state.idx
            st.rerun()
