# Devil-agent pilot, 2026-11-28

**Result:** on 181 claims the grounded assistant (Qwen3.8-27B) accepted 2 of 137 wrong or unsupported claims (1.5%) as true and rejected 1 of 44 true ones (2.3%, a bug in our own item). Exact-label accuracy was 95.0% with the wide corpus and 96.7% with PubMed abstracts only. Every miss was a reasoning miss; the right passage was always retrieved.

## Setup

- Devil: `openai:gpt-5.5`. Grounded model: `hf:Qwen/Qwen3.8-27B` (different families). No judge model: labels come from how each item was built.
- 20 evidence packs, one numeric result sentence each, sampled from the PubMed eval corpus (`--sources pubmed`, seed 11).
- Built 181 items: 36 paraphrases, 74 one-fact edits (number, direction, population, intervention), 20 wrong citations, 20 made-up PMIDs, 16 criteria-table thresholds (8 true, 8 changed), 15 untrusted claims. At build, code dropped 5 devil outputs (1 non-JSON reply, 2 paraphrases that changed a number, 2 population edits whose "changed from" words were not in the source).
- Same items judged twice: corpus A = PubMed eval corpus (2,015 abstracts) + criteria table; corpus B = A + new sources (15,808 passages: PMC and Europe PMC full text, trials, VA/DoD guidelines).

## Results

| Measure | A: PubMed only | B: wide corpus |
| --- | --- | --- |
| Exact-label accuracy | 96.7% | 95.0% |
| Supported vs not (binary) | 98.3% | 98.3% |
| Wrong claims accepted as supported | 1.5% (2 of 137) | 1.5% (2 of 137) |
| True claims rejected | 2.3% (1 of 44)* | 2.3% (1 of 44)* |
| Quote found in the cited passage | 99.3% | 99.3% |
| Misses: retrieval / reasoning | 0 / 6 | 0 / 9 |

\* That one item was a builder bug (below); with it fixed, no true claim was rejected.

| Item kind | n | A | B |
| --- | --- | --- | --- |
| Paraphrase (true) | 36 | 100% | 100% |
| Edit: number | 19 | 100% | 100% |
| Edit: direction | 19 | 100% | 94.7% |
| Edit: population | 17 | 94.1% | 94.1% |
| Edit: intervention | 19 | 78.9% | 68.4% |
| Wrong citation | 20 | 100% | 100% |
| Made-up PMID | 20 | 100% | 100% |
| Criteria thresholds | 16 | 93.8% | 93.8% |
| Untrusted claims | 15 | 100% | 100% |

## What we learned

1. **Swapped treatments are the weak spot.** Intervention edits (for example one drug or therapy for another) were accepted as true twice and called "unsupported" instead of "contradicted" 2 to 4 times. Next: hard negatives for intervention names and a prompt rule to compare the treatment named.
2. **The wider corpus did not hurt retrieval of the source**, but it added 3 reasoning misses in this pilot (96.7% to 95.0%, within noise at n = 181). These items came from abstracts, so this pilot cannot show the gain from full text; a pilot with `--sources pmc_oa,europe_pmc_oa,va_dod_guideline` is the next run.
3. **The grounded model caught a label bug.** Our builder wrote "PTSD symptoms must last at least 1 month"; the table says *more than* 1 month and the model called it contradicted. Fixed in `criteria_items()`.
4. **Format failures disappeared** after retrying empty replies with a larger token budget (17 of 181 replies were empty on the first attempt because the model spent its budget thinking).

Not shown here: the wrong statements themselves. They stay in `data/devil/` (gitignored); per-item scores are in the run folders.
