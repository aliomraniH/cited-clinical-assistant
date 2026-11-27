# Conversational multi-intent search

People write the way they talk: one message often holds several questions, lay words, and
context that changes the meaning. This layer sits in front of the retriever and answerer.

Plan and design notes: Claude Doc "Conversational multi-intent search: plan".

## Flow (`src/cca/converse.py`)

1. **Safety first.** A first-person self-harm mention returns crisis resources; nothing else runs.
2. **Interpret** (`src/cca/interpret.py`). The model splits the message into 1 to 4 readings, each with
   an intent type, a short search question, the user's own key sentences (checked to be verbatim),
   clinical terms, a confidence, and an `alternative` flag for either/or readings of the same words.
   Three samples are merged (self-consistency); unparseable samples are retried once.
3. **Search.** Each reading is searched separately (BM25 today; MedCPT hybrid is next).
   Readings that pull the same top-5 PMIDs are one question worded twice and are merged.
4. **Clarify gate.** One multiple-choice question, only when two readings are marked as alternatives of
   the same words, are close in confidence, and retrieve different evidence. Separate questions in one
   message are never "clarified"; they go to the confirm step.
5. **Confirm.** The assistant lists what it understood. The user can say yes, "drop 2", "only 1",
   correct it ("no, I meant ..."), or approve with a refinement ("yes, especially ..."), which is folded
   in without asking again.
6. **Answer.** One cited answer per confirmed reading (citations shown as PMIDs), then
   "Was this helpful? (yes / partly / no)".

## Feedback loop (`src/cca/tracking.py`)

Every turn is logged as an event (message, readings, queries, clarify, confirm, correction or
refinement, answer, feedback). When the session ends, each reading gets a value from -1 to 1:

| Signal | Weight |
|---|---|
| "Was this helpful?" yes / partly / no | +0.6 / +0.2 / -0.6 (x0.25 for a reading the user already confirmed: it rates the answer, not the understanding; also kept as `answer_feedback`) |
| Confirm accepted | +0.5 |
| Picked / not picked in a clarifying question | +0.3 / -0.3 |
| Follow-up that builds on the answer | +0.3 |
| Dropped by the user | -0.7 |
| Corrected by the user | -1.0 |
| User rephrased the same request | -0.4 |

Query changes across turns are stored as before/after pairs with the user's words as the reason.

Scored (phrasing -> reading) pairs go to the **phrasing memory**. A pair is used once its value is at
least 0.5 in size or it was seen twice. On a new message the closest past phrasings are given to the
interpreter as examples ("people who said X meant Y" / "did not mean Y"). Text is de-identified before
it is stored, and only `roleplay` and `test` sessions teach the memory until the clinician review is done;
live sessions read it but never write it.

## Interfaces and stored sessions

Conversation state is saved after every turn (`src/cca/store.py`), so the chatbot, the HTTP API, the MCP
server or another replica can all continue the same conversation by its `session_id`, the way a hosted chat
API keeps stored conversations.

- Python: `Assistant.converse(message, session_id)`, `get_conversation(session_id)`, `end_conversation(session_id)`.
- HTTP: `POST /v1/converse {message, session_id?}`, `GET /v1/converse/{session_id}`, `POST /v1/converse/end {session_id}`.
- MCP: tools `converse`, `get_conversation`, `end_conversation`.
- Chatbot (`app.py`): the chat tab now runs conversational search; the session id rides in a hidden comment.

| `CCA_SESSION_STORE` | Where state lives |
|---|---|
| `file` (default) | `data/sessions/*.json`, one machine |
| `supabase` | table `conversation_state` (no anon read); needs `SUPABASE_URL` and, outside the cloud proxy, `SUPABASE_KEY` (secret key) |
| `memory` | this process only (tests) |

**Running the app on its own (Hugging Face Space):** add Space secrets `SUPABASE_KEY`, and `ANTHROPIC_API_KEY`
and/or `OPENAI_API_KEY` if a closed model should answer; set variables `CCA_SESSION_STORE=supabase`,
`SUPABASE_URL`, and `CCA_MODEL` (`hf:Qwen/Qwen3.8-27B`, `anthropic:claude-sonnet-5-5` or `openai:gpt-5.5`).
No Claude session is needed for the app to run.

## Role-play eval (`src/cca/roleplay.py`, `scripts/run_roleplay.py`)

A simulator model plays the person from a hidden scenario card (`eval/scenarios.jsonl`: persona,
style, hidden intents with keywords, planned corrections, ambiguity notes, safety cases). The
Conversation answers; a judge from a third model family grades understanding, clarification,
answer relevance and safety. `--swap` exchanges simulator and judge on every other scenario.

```
python scripts/fetch_eval_corpus.py
python scripts/run_roleplay.py --system hf:Qwen/Qwen3.8-27B --sim openai:gpt-5.5 \
    --judge hf:deepseek-ai/DeepSeek-V4-Flash
python scripts/run_roleplay.py --sim anthropic:claude-sonnet-5-5 --judge openai:gpt-5.5 --swap
python scripts/upload_roleplay.py eval/roleplay_runs/<run_id> --memory data/phrasing_memory.jsonl
```

Keys: `CCA_ANTHROPIC_KEY` environment variable in cloud sessions (the platform reserves `ANTHROPIC_API_KEY`), `ANTHROPIC_API_KEY` as a Space secret, for `anthropic:`; the OpenAI Network secret for
`openai:`; `hf:` uses the Hugging Face Network secret. Supabase tables: `db/schema_conversations.sql`.
