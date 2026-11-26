# Cited clinical assistant

A learning project: an assistant that answers clinical questions about generalized anxiety disorder, depression, OCD and PTSD **only from retrieved PubMed literature**, cites every claim, and says when the evidence is not enough. It is built to practice dataset, model, behavior and application metrics end to end.

**Research demo, not medical advice.** No patient or client data is used: only public PubMed records and synthetic test questions.

## Status

| Part | State |
| --- | --- |
| Daily PubMed pull, cleaning, data audit | Working (`scripts/daily_pull.py`) |
| Retrieval index (BM25, then MedCPT) | Next |
| Answerer with citations (model chosen by eval) | Next |
| Eval harness and live site | Planned |

## Daily pull

```bash
pip install -e ".[dev]"
python scripts/daily_pull.py --days 1        # writes data/snapshots/<date>.jsonl and data/audits/<date>.json
pytest
```

Topics and their PubMed queries live in `config/topics.json`. Set `NCBI_API_KEY` to raise the NCBI rate limit from 3 to 10 requests a second. Snapshots are kept out of git because abstracts can carry publisher copyright.

Each run tags every record with its source, PMID, URL, fetch time and terms note, removes cross-topic and near-duplicate copies, and runs deterministic checks: schema validity, missing fields, duplicates, topic balance, a PII pattern scan, and drift against the previous snapshot. A PII hit or schema failure exits with code 2.
