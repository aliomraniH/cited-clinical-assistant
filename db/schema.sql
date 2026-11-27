-- Cited clinical assistant: run registry and data snapshots.
-- Public PubMed data and synthetic questions only; no patient data.
create table if not exists public.runs (
  run_id text primary key,
  created_at timestamptz not null default now(),
  model text not null,
  judge_model text,
  retriever text,
  k int,
  corpus_records int,
  items int,
  summary jsonb not null
);
create table if not exists public.run_items (
  run_id text not null references public.runs(run_id) on delete cascade,
  item_id text not null,
  topic text,
  answerable boolean,
  abstained boolean,
  error text,
  gold_rank int,
  latency_s real,
  answer text,
  metrics jsonb,
  judge jsonb,
  primary key (run_id, item_id)
);
create table if not exists public.snapshots (
  snapshot_date date primary key,
  created_at timestamptz not null default now(),
  records int,
  audit jsonb
);
alter table public.runs enable row level security;
alter table public.run_items enable row level security;
alter table public.snapshots enable row level security;
-- The dashboard may read with the publishable key; only the secret key (which bypasses RLS) can write.
drop policy if exists "public read" on public.runs;
create policy "public read" on public.runs for select to anon, authenticated using (true);
drop policy if exists "public read" on public.run_items;
create policy "public read" on public.run_items for select to anon, authenticated using (true);
drop policy if exists "public read" on public.snapshots;
create policy "public read" on public.snapshots for select to anon, authenticated using (true);
-- New Supabase projects do not grant table access to API roles by default.
grant select on public.runs, public.run_items, public.snapshots to anon, authenticated;
grant all on public.runs, public.run_items, public.snapshots to service_role;
