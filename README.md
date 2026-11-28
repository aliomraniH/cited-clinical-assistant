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
| Full text, trials and VA/DoD guidelines | Working (`scripts/pull_sources.py`) |
| Criteria cases and devil test | Pilot (`scripts/run_devil.py`) |

## Interfaces

The assistant is reachable four ways, all backed by `src/cca/service.py` with the same safety rules (crisis routing, no diagnosis). Details in [docs/interfaces.md](docs/interfaces.md).

```bash
pip install -e ".[dev,serve,overlap]"
python app.py                                   # chatbot at :7860, REST at /v1, MCP at /mcp/, agent card at /.well-known/agent-card.json
python -m cca.interfaces.mcp_server             # MCP over stdio for Claude Desktop / Claude Code
```

1. **MCP server** that can consult other MCP servers (allowlisted in `config/mcp_peers.json`) for decision briefs.
2. **Chatbot** (Gradio), deployed from GitHub to a Hugging Face Space by `.github/workflows/deploy-space.yml`.
3. **HTTP API** with OpenAPI spec and an A2A-style agent card, for other agents.
4. **Architecture Coach**: a Claude skill plus a [coaching page](https://claude.ai/artifact/4qhFi7M1ofhXWVCPc4Bu5z) for interview prep.

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

## More sources

`scripts/pull_sources.py` adds four sources beside the PubMed abstracts, all as passage rows that BM25 and the answerer already understand (config in `config/sources.json`, data in `data/sources/`, kept out of git):

| Source | Route | What is kept |
| --- | --- | --- |
| PMC open-access full text | NCBI E-utilities (`db=pmc`) | Articles whose own licence statement is CC BY or CC0; others are dropped and counted |
| Europe PMC open-access full text | Europe PMC REST API | Same licence rule; articles already pulled from PMC are skipped |
| ClinicalTrials.gov | API v2 | Summary, eligibility and primary outcomes per study |
| VA/DoD guidelines (PTSD 2023, MDD 2022, SUD 2021) | PDF from healthquality.va.gov | Page-level passages, cited as guideline + page |

```bash
pip install -e ".[dev,sources]"
python scripts/pull_sources.py --per-topic 30           # all four sources
python scripts/ask.py "Is trauma-focused therapy first line for PTSD?" --snapshots data/eval_corpus data/sources --retrieve-only
```

Full-text passages keep the paper's PMID as their citation, so several passages can cite one paper; trials cite their NCT id and guidelines cite `VADoD-PTSD-2023 p.45`.

## Criteria cases and the devil test

- `config/criteria.json`: diagnostic criteria for GAD, depression, PTSD, OCD (plus ADHD, substance use and eating disorders for overlap) **in our own words**, keyed by DSM-5-TR criterion letter. Not DSM text. Draft for clinician review.
- `scripts/make_criteria_cases.py` writes `eval/criteria_cases.jsonl`: 20 synthetic vignettes that fit or miss one criterion (duration, symptom count, stressor, substance cause, or a better explanation such as PTSD instead of GAD). Code re-derives every label from the case facts and the thresholds, and builds the "why it fits / why not" text from the table. `--corpus` attaches the top passages from our sources as evidence.
- `scripts/run_devil.py` runs the adversarial test. A model from another family (default `openai:gpt-5.5`) rewrites sentences from our sources into faithful paraphrases and one-fact edits; code builds wrong citations, made-up PMIDs, threshold changes in the criteria table, and loads `eval/untrusted_claims.jsonl`. The grounded assistant (default `hf:Qwen/Qwen3.8-27B`) judges each claim against our sources as supported, contradicted or unsupported, quoting the passage it used.

The devil never sets a label: each label comes from how the item was built, and an item whose edit cannot be confirmed by code is dropped. Every miss is tagged *retrieval* (the source passage was never shown) or *reasoning* (shown but misjudged). Wrong claims are flagged `synthetic_false`, written only under `data/devil/`, and `load_snapshots()` refuses such rows, so they cannot reach the index or the chatbot. `eval/devil_runs/` keeps summaries and per-item scores without the wrong statements.

```bash
python scripts/make_criteria_cases.py --corpus data/eval_corpus data/sources
python scripts/run_devil.py --corpus data/eval_corpus data/sources --packs 20 --sources pubmed
python scripts/run_devil.py --corpus data/eval_corpus --items data/devil/<run>/items.jsonl   # same claims, smaller corpus
```

## Eval

```bash
python scripts/fetch_eval_corpus.py                 # rebuilds the frozen 2,015-record corpus from eval/corpus_pmids.txt
python scripts/run_eval.py --retrieval-only         # retrieval and gating metrics, no model calls
python scripts/run_eval.py --model Qwen/Qwen3.5-9B --base-url https://router.huggingface.co/v1 \
    --judge-model openai/gpt-oss-120b               # full run; results/<run_id>/summary.json and items.jsonl
```

`eval/questions.jsonl` has 21 answerable questions, each written from one paper in the frozen corpus with its key facts, and 8 unanswerable ones: off-topic questions, or in-domain questions the corpus does not cover. Metrics follow the eval-run skill. Code grades recall@10, abstention, gold-paper citation, uncited sentences and latency. A judge from a different model family grades citation support and key-fact coverage, and the script refuses a judge from the same family as the model under test.
