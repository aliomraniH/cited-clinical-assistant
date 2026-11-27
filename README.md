# Cited clinical assistant

A learning project: an assistant that answers clinical questions about generalized anxiety disorder, depression, OCD and PTSD **only from retrieved PubMed literature**, cites every claim, and says when the evidence is not enough. It is built to practice dataset, model, behavior and application metrics end to end.

**Research demo, not medical advice.** No patient or client data is used: only public PubMed records and synthetic test questions.

## Status

| Part | State |
| --- | --- |
| Daily PubMed pull, cleaning, data audit | Working (`scripts/daily_pull.py`) |
| Retrieval (BM25 baseline; MedCPT next) | Working (`src/cca/index.py`) |
| Answerer with checked citations | Working with any OpenAI-compatible endpoint (`scripts/ask.py`) |
| Eval harness and live site | Planned |

## Daily pull

```bash
pip install -e ".[dev]"
python scripts/daily_pull.py --days 1        # writes data/snapshots/<date>.jsonl and data/audits/<date>.json
pytest
```

Topics and their PubMed queries live in `config/topics.json`. Set `NCBI_API_KEY` to raise the NCBI rate limit from 3 to 10 requests a second. Snapshots are kept out of git because abstracts can carry publisher copyright.

Each run tags every record with its source, PMID, URL, fetch time and terms note, removes cross-topic and near-duplicate copies, and runs deterministic checks: schema validity, missing fields, duplicates, topic balance, a PII pattern scan, and drift against the previous snapshot. A PII hit or schema failure exits with code 2.

## Ask a question

```bash
python scripts/ask.py "Does prolonged exposure therapy reduce PTSD symptoms?" --retrieve-only   # passages only
ollama pull qwen3.5:9b                                                                          # or any candidate model
python scripts/ask.py "Does prolonged exposure therapy reduce PTSD symptoms?" --model qwen3.5:9b
```

The answerer sends the top passages, numbered, to the model and requires a citation such as `[2]` on every sentence. If the passages do not cover the question, the model must reply `INSUFFICIENT_EVIDENCE`. Questions whose best BM25 score is below `--min-score` are refused before the model is called. Code then checks each answer for uncited sentences and citations to passages that do not exist. Whether a cited passage really supports its sentence is left to the cross-family judge in the eval harness.

Point `--base-url` (or `CCA_BASE_URL`, plus `CCA_API_KEY` if needed) at any OpenAI-compatible server, such as Ollama, vLLM, llama.cpp or Hugging Face Inference Providers, to swap models without changing code.
