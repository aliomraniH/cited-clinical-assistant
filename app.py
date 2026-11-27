"""Interface 2: standalone chatbot (Gradio), served together with the HTTP API and MCP endpoint.

  python app.py            # http://localhost:7860  (UI at /, API at /v1, MCP at /mcp, docs at /docs)

On Hugging Face Spaces this file is the entry point. Set Space secrets:
  CCA_API_KEY   Hugging Face token with Inference Providers permission (for the answer model)
  CCA_API_TOKEN optional bearer token that protects /v1 and /mcp for other agents
The first start rebuilds the 2,015-record PubMed corpus (about a minute).
"""

from __future__ import annotations

import html
import math
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ.setdefault("CCA_BASE_URL", "https://router.huggingface.co/v1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from cca.corpus import ensure_corpus  # noqa: E402
from cca.interfaces.http_api import create_app  # noqa: E402
from cca.safety import CRISIS_RESOURCES, DISCLAIMER  # noqa: E402
from cca.service import get_assistant  # noqa: E402

A = get_assistant()

EXAMPLES = [
    "Does exposure and response prevention reduce OCD symptoms in adolescents?",
    "What is the evidence for trauma-focused CBT in veterans with PTSD?",
    "How common is comorbid depression in generalized anxiety disorder?",
    "Do SSRIs help PTSD-related nightmares?",
]
OVERLAP_EXAMPLE = ("For the past year she worries about many things and can't switch off, sleeps badly, feels tired "
                   "and has lost interest in seeing friends. She has trouble concentrating at work.")


def _resources_md() -> str:
    return "\n".join(f"- **{r['name']}** {r['contact']} {r['url']}".rstrip() for r in CRISIS_RESOURCES)


def format_answer(r: dict) -> str:
    if r["status"] == "crisis":
        return f"{r['answer']}\n\n{_resources_md()}"
    if r["status"] == "diagnosis_request":
        return r["answer"]
    out = [r["answer"]]
    if r.get("citations"):
        out.append("\n**Sources**")
        out += [f"[{c['n']}] [{html.escape(c.get('title') or c['pmid'])}]({c['url']}) PMID {c['pmid']}"
                for c in r["citations"]]
    ch = r.get("checks") or {}
    if ch and not ch.get("abstained"):
        rate = ch.get("cited_sentence_rate")
        out.append(f"\n<sub>Checks: {ch.get('sentences')} sentences, "
                   f"{'' if rate is None else f'{rate:.0%} cited, '}{ch.get('invalid_citations')} invalid citations. "
                   f"Model {r.get('model')}, {r.get('latency_s')} s.</sub>")
    if r.get("issues"):
        out.append("<sub>Flags: " + "; ".join(html.escape(i) for i in r["issues"][:3]) + "</sub>")
    if (r.get("safety") or {}).get("kind") == "risk_topic":
        out.append(f"\n<sub>{r['safety']['message']}</sub>")
    out.append(f"\n<sub>{DISCLAIMER}</sub>")
    return "\n".join(out)


_SID = re.compile(r"<!-- sid:([A-Za-z0-9_.-]+) -->")


def _session_from(history) -> str | None:
    """The conversation id rides in a hidden comment on the assistant's last reply, so the chat needs no
    server-side UI state; the conversation itself is stored (store.py) and also reachable from the API."""
    for turn in reversed(history or []):
        content = turn.get("content") if isinstance(turn, dict) else (turn[1] if len(turn) > 1 else "")
        m = _SID.search(str(content or ""))
        if m:
            return m.group(1)
    return None


def chat(message: str, history):
    try:
        r = A.converse(message, _session_from(history))
    except Exception as e:
        return f"Something went wrong reaching the model ({type(e).__name__}). Try again in a minute."
    text = r["reply"]
    if r.get("state") == "crisis":
        text += "\n\n" + _resources_md()
    return f"{text}\n\n<!-- sid:{r['session_id']} -->"


def graph_svg(res: dict) -> str:
    """Conditions as nodes sized by score; an edge for each shared symptom between flagged conditions."""
    agent = A.overlap_engine()
    labels = {c: v["label"] for c, v in agent.conditions.items()}
    scores = res.get("scores") or {}
    nodes = [c for c in labels if scores.get(c, 0) > 0]
    if not nodes:
        return ""
    w, h, cx, cy, rad = 560, 420, 280, 210, 140
    pos = {c: (cx + rad * math.cos(2 * math.pi * i / len(nodes) - math.pi / 2),
               cy + rad * math.sin(2 * math.pi * i / len(nodes) - math.pi / 2)) for i, c in enumerate(nodes)}
    flagged = set(res.get("alert_conditions", [])) | set(res.get("consider", []))
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Overlap graph" '
             'style="max-width:560px;font-family:sans-serif">']
    for s in res.get("shared_symptoms", []):
        conds = [c for c in agent.symptoms[s]["conditions"] if c in pos and c in flagged]
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                (x1, y1), (x2, y2) = pos[conds[i]], pos[conds[j]]
                parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#d97706" '
                             f'stroke-width="2" opacity="0.7"><title>{html.escape(agent.symptoms[s]["label"])}</title></line>')
    for c, (x, y) in pos.items():
        r = 18 + 30 * scores[c]
        fill = "#d97706" if c in res.get("alert_conditions", []) else ("#2563eb" if c in flagged else "#94a3b8")
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" fill="{fill}" opacity="0.85"/>'
                     f'<text x="{x:.0f}" y="{(y - r - 6) if y < cy - 1 else (y + r + 14):.0f}" text-anchor="middle" '
                     f'font-size="12" fill="currentColor">'
                     f'{html.escape(labels[c])} {scores[c]:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def run_overlap(text: str):
    if not text or len(text.strip()) < 10:
        return "Describe the symptoms in a sentence or two (no names or identifying details).", ""
    try:
        r = A.overlap(text)
    except Exception as e:
        return f"Overlap check failed ({type(e).__name__}).", ""
    if r["status"] == "crisis":
        return f"{r['message']}\n\n{_resources_md()}", ""
    agent = A.overlap_engine()
    md = [f"### {'Overlap alert' if r['alert'] else 'No overlap alert'}"]
    if r.get("message"):
        md.append(r["message"])
    if r.get("questions"):
        md.append("\n**Distinguishing features to ask about**")
        for q in r["questions"]:
            head, _, rest = q.partition(":")
            items = "".join(f"<li>{html.escape(x.strip())}</li>" for x in rest.split(";") if x.strip())
            md.append(f"<details><summary>{html.escape(head)}</summary><ul>{items}</ul></details>")
    if r.get("consider"):
        md.append("\n**Also consider:** " + ", ".join(agent.conditions[c]["label"] for c in r["consider"]))
    matched = [m for m in r.get("matches", []) if m["symptom"]]
    if matched:
        md.append("\n**Matched symptoms**\n" + "\n".join(
            f"- \"{html.escape(m['phrase'])}\" → {agent.symptoms[m['symptom']]['label']} ({m['similarity']:.2f})"
            for m in matched))
    md.append(f"\n<sub>Not a diagnosis. {DISCLAIMER}</sub>")
    return "\n".join(md), graph_svg(r)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cited clinical assistant") as demo:
        gr.Markdown("# Cited clinical assistant\nAnswers questions about generalized anxiety, depression, OCD and "
                    "PTSD only from PubMed abstracts, with a citation on every sentence. "
                    f"**{DISCLAIMER}** Do not enter identifying details.")
        with gr.Tab("Ask the literature"):
            gr.ChatInterface(chat, examples=EXAMPLES, cache_examples=False)
        with gr.Tab("Symptom overlap check"):
            inp = gr.Textbox(label="De-identified symptom description", lines=5, value=OVERLAP_EXAMPLE)
            btn = gr.Button("Check overlap", variant="primary")
            out_graph, out_md = gr.HTML(), gr.Markdown()
            btn.click(run_overlap, inp, [out_md, out_graph])
        with gr.Tab("For agents"):
            gr.Markdown("This app also serves:\n\n- REST API: `POST /v1/ask`, `/v1/search`, `/v1/overlap`, `/v1/brief` "
                        "(spec at `/openapi.json`, docs at `/docs`)\n- MCP server (streamable HTTP): `/mcp/`\n"
                        "- Agent card: `/.well-known/agent-card.json`\n\nSource: "
                        "[github.com/aliomraniH/cited-clinical-assistant](https://github.com/aliomraniH/cited-clinical-assistant)")
    return demo


def create() -> "object":
    api = create_app(A)
    return gr.mount_gradio_app(api, build_ui(), path="/")


if __name__ == "__main__":
    n = ensure_corpus(Path(A.corpus_dir))
    print(f"corpus: {n} records in {A.corpus_dir}")
    uvicorn.run(create(), host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
