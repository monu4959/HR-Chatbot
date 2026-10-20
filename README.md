# HR-Chatbot

[![CI](https://github.com/monu4959/HR-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/monu4959/HR-Chatbot/actions/workflows/ci.yml)
[![CD](https://github.com/monu4959/HR-Chatbot/actions/workflows/cd.yml/badge.svg)](https://github.com/monu4959/HR-Chatbot/actions/workflows/cd.yml)

A chatbot that answers questions about 1,000 employee recognition messages: counts and distributions,
messages on a theme, people's skill profiles and possible successors, culture insights, charts, and who
recognizes whom across teams. It runs on a local model (qwen2.5:7b via Ollama) and comes with an
evaluation suite, a validated answer grader, caching and a model router.

## Run it

You need [Ollama](https://ollama.com) with the model pulled: `ollama pull qwen2.5:7b`.

**With Docker** (the image every passing commit on `main` publishes):

```bash
docker compose up
```

Then open the UI at http://localhost:8501 (the API is on http://localhost:8000). To build from your
own checkout instead of pulling the published image, run `docker compose up --build`.

**Without Docker:**

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                        # then fill in what you use
uvicorn api.main:app --port 8000            # API
streamlit run frontend/app.py               # UI, in a second terminal
```

## How it works

A router model picks one of six tools for each question (pandas analytics, semantic search, profiles,
culture insights, charts, recognition graph). A model router then decides how much model work the answer
needs: exact results are formatted directly from the tool output, and the rest are written by the model
and checked by a faithfulness grader before they are shown. Questions are traced in LangSmith, and model
calls and whole answers are cached.

## Results

| Measure | Result |
|---|---|
| Accuracy on the 49 dev+test cases | 69.4% → 98.0% after the tool fixes and router |
| Accuracy on 35 held-out cases (never tuned on) | about 74% |
| Original grader vs human labels (Cohen's κ) | 0.00 |
| Validated graders, Claude Haiku (κ, holdout) | correctness 0.90, faithfulness 0.81 |
| Mean time per question with the model router | 21.6 s → 7.4 s |

## Evals and CI/CD

```bash
python -m pytest tests -q                    # 161 tests, no model calls
python -m evals.run --split all --gate       # full eval + regression gate (Haiku judge as second scorer)
python -m evals.grader_validation.validate_grader --split holdout
```

- **CI** (`.github/workflows/ci.yml`): every push runs the tests and checks that the gold answers still match the data.
- **CD** (`.github/workflows/cd.yml`): when CI passes on `main`, the Docker image is built, smoke-tested and
  published to `ghcr.io/monu4959/hr-chatbot`, tagged `latest` and with the commit SHA.
- **LLM eval** (`.github/workflows/llm-eval.yml`): the full model eval and gate, started by hand.

## Layout

| Path | Contents |
|---|---|
| `agent/` | LangGraph agent, model factory, grader, cache, model router, prompts |
| `tools/` | the six tools |
| `api/`, `frontend/` | FastAPI backend, Streamlit UI |
| `data/annotated_results.csv` | the dataset (1,000 labelled recognition messages) |
| `labelling/` | how the dataset was labelled: taxonomy creation and annotation ([README](labelling/README.md)) |
| `evals/` | test set, eval runner, graders, grader validation, regression gate |
| `tests/` | unit tests |
