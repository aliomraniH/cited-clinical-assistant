"""Generate labelled synthetic vignettes for the overlap eval.

Labels come from the spec (which symptoms we ask the writer to include), not from the
writer. The writer is a different family (gpt-oss) from the extractor under test (Qwen).
Synthetic only: no real patient text.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cca.answer import ChatClient  # noqa: E402
from cca.overlap import MAP_PATH, RISK  # noqa: E402

CORE = ["gad", "depression", "ptsd", "ocd"]
WRITER_PROMPT = """You write short synthetic clinical vignettes for testing software. Write 3 to 5 sentences, third person,
about a fictional adult, as a clinician might summarise a first appointment. Describe every listed feature in plain
everyday words (do not copy the wording). Add no other symptoms. Never name a diagnosis or disorder, and never
mention suicide or self-harm. Reply with the vignette text only."""
BANNED = re.compile(r"\b(PTSD|OCD|GAD|ADHD|depression|depressive|anxiety disorder|obsessive|compulsive|post-?traumatic|disorder)\b", re.I)


def build_specs(smap: dict, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    syms = smap["symptoms"]
    spec_of = {c: [s for s in syms if s["conditions"] == [c]] for c in CORE}
    shared_of = {c: [s for s in syms if c in s["conditions"] and len(s["conditions"]) > 1] for c in CORE}
    dist = smap["distinguishing"]
    specs = []
    for i in range(30):  # single
        c = CORE[i % 4]
        feats = rng.sample(spec_of[c], 3) + rng.sample(shared_of[c], 1)
        specs.append({"kind": "single", "conditions": [c], "features": [f["label"] for f in feats], "symptoms": [f["id"] for f in feats]})
    pairs = list(itertools.combinations(CORE, 2))
    for i in range(30):  # comorbid
        a, b = pairs[i % len(pairs)]
        feats = rng.sample(spec_of[a], 2) + rng.sample(spec_of[b], 2)
        common = [s for s in shared_of[a] if b in s["conditions"]]
        if common:
            feats.append(rng.choice(common))
        specs.append({"kind": "comorbid", "conditions": [a, b], "features": [f["label"] for f in feats], "symptoms": [f["id"] for f in feats]})
    for i in range(30):  # trap: mostly shared symptoms, one specific
        c = CORE[i % 4]
        feats = rng.sample(spec_of[c], 1) + rng.sample(shared_of[c], min(3, len(shared_of[c])))
        specs.append({"kind": "trap", "conditions": [c], "features": [f["label"] for f in feats], "symptoms": [f["id"] for f in feats]})
    for i in range(30):  # distinguishing: shared symptoms described with this condition's telling feature
        c = CORE[i % 4]
        opts = [s for s in shared_of[c] if c in dist.get(s["id"], {})]
        feats = rng.sample(opts, min(3, len(opts)))
        specs.append({"kind": "distinguishing", "conditions": [c],
                      "features": [f"{f['label']} ({dist[f['id']][c]})" for f in feats], "symptoms": [f["id"] for f in feats]})
    for n, s in enumerate(specs, 1):
        s["id"] = f"v{n:03d}"
    return specs


def write(spec: dict, client: ChatClient) -> dict:
    feats = "\n".join(f"- {f}" for f in spec["features"])
    for _ in range(3):
        text = client.complete([{"role": "system", "content": WRITER_PROMPT}, {"role": "user", "content": f"Features:\n{feats}"}],
                               temperature=0.7, max_tokens=2048).strip()
        if text and not BANNED.search(text) and not RISK.search(text):
            return {**spec, "text": text, "writer": client.model}
    return {**spec, "text": "", "writer": client.model, "error": "no clean vignette after 3 tries"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--out", type=Path, default=Path("eval/vignettes.jsonl"))
    args = ap.parse_args()
    smap = json.loads(MAP_PATH.read_text())
    client = ChatClient(args.model, base_url=args.base_url)
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda s: write(s, client), build_specs(smap)))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bad = sum(1 for r in rows if not r["text"])
    print(f"wrote {len(rows)} vignettes to {args.out}, {bad} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
