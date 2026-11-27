---
title: Cited Clinical Assistant
emoji: 📚
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Cited PubMed answers + symptom overlap check (research demo)
---

# Cited clinical assistant (research demo, not medical advice)

Answers questions about generalized anxiety, depression, OCD and PTSD only from PubMed abstracts, with a
citation on every sentence, and checks de-identified symptom descriptions for overlap between conditions.

Deployed automatically from https://github.com/aliomraniH/cited-clinical-assistant (do not edit here; changes
are overwritten on the next deploy). The same app serves a REST API at `/v1`, an MCP server at `/mcp/` and an
agent card at `/.well-known/agent-card.json`.
