-- Conversational search: session logs, role-play runs and the phrasing memory (feedback loop).
-- Role-play and test sessions only until the clinician review; text is de-identified before upload.
create table if not exists public.conversations (
  session_id text primary key,
  created_at timestamptz not null default now(),
  source text not null,              -- roleplay | test | live
  run_id text,                       -- role-play run, if any
  scenario text,
  final_state text,
  metrics jsonb,
  judge jsonb,
  query_changes jsonb,
  readings jsonb
);
create table if not exists public.conversation_events (
  session_id text not null references public.conversations(session_id) on delete cascade,
  seq int not null,
  turn int not null,
  role text not null,
  kind text not null,
  payload jsonb not null,
  ts timestamptz,
  primary key (session_id, seq)
);
create table if not exists public.phrasing_memory (
  phrasing text not null,
  paraphrase text not null,
  intent_type text,
  terms jsonb,
  value real not null,
  seen int not null default 1,
  sessions jsonb,
  reasons jsonb,
  source text,
  version text not null,
  updated_at timestamptz not null default now(),
  primary key (version, phrasing, paraphrase)
);
alter table public.conversations enable row level security;
alter table public.conversation_events enable row level security;
alter table public.phrasing_memory enable row level security;
drop policy if exists "public read" on public.conversations;
create policy "public read" on public.conversations for select to anon, authenticated using (true);
drop policy if exists "public read" on public.phrasing_memory;
create policy "public read" on public.phrasing_memory for select to anon, authenticated using (true);
-- Event logs hold full transcripts: no anon read.
grant select on public.conversations, public.phrasing_memory to anon, authenticated;
grant all on public.conversations, public.conversation_events, public.phrasing_memory to service_role;

-- Live conversation state, so any process (chatbot, API, MCP server) can resume a session by id.
-- Holds transcripts: no anon read; only the secret key (service_role) reads and writes it.
create table if not exists public.conversation_state (
  session_id text primary key,
  source text not null,
  state jsonb not null,
  updated_at timestamptz not null default now()
);
alter table public.conversation_state enable row level security;
grant all on public.conversation_state to service_role;
