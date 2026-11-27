# Role-play baseline: conversational multi-intent search (2026-11-27)

Run `20261127T185753Z-roleplay`. System Qwen3.8-27B (HF Inference Providers), 3 interpreter samples per turn.
Simulator GPT-5.5 (OpenAI) playing the person from 12 hidden scenario cards. Judge DeepSeek-V4-Flash
(a third family, so neither the system's nor the simulator's family grades itself). Corpus: the frozen
2,015-record eval corpus, BM25. Transcripts, events and the phrasing memory are in Supabase
(`conversations`, `conversation_events`, `phrasing_memory`).

| Metric | Value |
|---|---|
| Hidden intents covered by confirmed readings | 1.00 (11 non-crisis scenarios) |
| Clarify decision correct (asked iff the card is ambiguous) | 0.82 (missed both ambiguous cards: asked via confirm instead) |
| Planned corrections handled | 2 of 2 |
| Crisis routed / diagnosis named | 1 of 1 / 0 |
| Mean user turns | 3.25 |
| Person's own feedback | 7 partly, 4 no, 0 yes |
| Judge: understood / clarification / answer relevance / safety (1-5) | 4.2 / 4.7 / 2.8 / 5.0 |
| Phrasing memory after run | 23 rows, 16 promoted |

**Reading:** the conversation layer works: it finds every hidden intent, takes corrections and
refinements, and keeps safety. Answers are the weak part: 15 of 21 answer sections abstained
("not enough evidence in the indexed PubMed records"). The eval corpus was built for the 29 answer-eval
questions, and BM25 matches poorly on paraphrased questions. Next: MedCPT hybrid retrieval and the
daily-snapshot corpus, then rerun this set.

**Bugs found by earlier runs (fixed before this one):** empty interpreter output from the reasoning model
(token budget), near-duplicate readings, clarifying questions for separate asks, a refinement ("yes, and
also ...") dropping earlier topics, answer-quality feedback scored as misunderstanding, and reworded readings
counted as corrected.

**Still open:** both ambiguous cards (sc003, sc010) went to the confirm step without a clarifying question,
because the model did not mark the readings as either/or; the confirm step caught the meaning anyway.
Claude as simulator waits on a fresh session (the key is CCA_ANTHROPIC_KEY).
