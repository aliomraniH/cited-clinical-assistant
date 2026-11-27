# Overlap agent baseline, 2026-11-27

120 synthetic vignettes (eval/vignettes.jsonl), 30 of each kind: single condition, true comorbidity (two conditions),
overlap traps (mostly shared symptoms, one condition), and vignettes that state the distinguishing features.
Written by openai/gpt-oss-120b from specs; labels come from the spec, not the writer. Odd ids = dev (thresholds tuned
by grid search), even ids = test (reported). A positive is a comorbid vignette; the alert should name both conditions.

| Extractor | Embedder | Alert precision | Alert recall (pair found) | F1 | False alarms: single / trap / distinguishing | Top condition correct |
|---|---|---|---|---|---|---|
| Clause splitter (no model) | SapBERT | 40.0% | 13.3% | 0.20 | 0% / 13.3% / 6.7% | 77.8% |
| Qwen3.5 9B | SapBERT | 56.2% | 60.0% | 0.58 | 0% / 26.7% / 20.0% | 77.8% |
| Qwen3.5 9B | Qwen3-Embedding 0.6B | 46.7% | 46.7% | 0.47 | 6.7% / 26.7% / 20.0% | 82.2% |

Test half: 60 vignettes, 15 per kind, so one vignette moves a rate by 6.7 points. Treat differences under ~15 points as noise.

Findings
- An LLM extractor is needed: splitting clauses finds the pair in 13% of comorbid cases versus 60% with Qwen3.5 9B.
- The trap problem shows up as designed: about 1 in 4 trap vignettes still raises an alert.
- Qwen3-Embedding beat SapBERT on dev (F1 0.79 vs 0.68) and lost on test (0.47 vs 0.58): tuning 240 settings on 60 vignettes overfits.
- 20 of 120 Qwen extractions still fell back to clause splitting (unparseable output).
- Two bugs found by this eval: Qwen's bracket-per-line output was not parsed (115/120 fell back), and scores were normalised by each condition's full symptom count, penalising PTSD and OCD.

Next ideas: more vignettes (tuning set separate from test), use the distinguishing notes in scoring, a small reranker for traps, clinician review of the map.
