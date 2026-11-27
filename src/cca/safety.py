"""Safety rules every interface applies before and after the assistant runs.

1. Risk gate (same pattern as the overlap agent): a first-person mention of suicide or
   self-harm ("I want to end my life") gets crisis resources and nothing else. A topical
   mention in a literature question ("suicide risk in PTSD") is answered, with the crisis
   resources attached. Case descriptions sent to the overlap check are always gated.
2. No diagnosis: requests to diagnose a person are declined; the assistant answers about
   literature and points to distinguishing features, never "you have X".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .overlap import RISK as _OVERLAP_RISK

# Broader than the overlap agent's pattern, which misses "killing myself", "want to die", "hurt myself".
RISK = re.compile(
    _OVERLAP_RISK.pattern.removesuffix(r")\b") + r"|kill(?:ing|ed)? (?:my|him|her|them)sel(?:f|ves)|(?:want|wanting|wish) to die|"
    r"better off dead|hurt(?:ing)? (?:my|him|her|them)self|cut(?:ting)? (?:my|him|her|them)self|take (?:my|his|her) own life)\b",
    re.I,
)

DISCLAIMER = "Research demo, not medical advice."

CRISIS_RESOURCES = [
    {"name": "988 Suicide and Crisis Lifeline (US)", "contact": "call or text 988", "url": "https://988lifeline.org"},
    {"name": "Crisis Text Line (US, UK, CA, IE)", "contact": "text HOME to 741741 (US)", "url": "https://www.crisistextline.org"},
    {"name": "Find a helpline (international)", "contact": "", "url": "https://findahelpline.com"},
    {"name": "Emergency services", "contact": "local emergency number", "url": ""},
]

CRISIS_MESSAGE = ("It sounds like suicide or self-harm may be on your mind. You don't have to handle this alone. "
                  "In the US, call or text 988 (Suicide and Crisis Lifeline); elsewhere, contact local emergency "
                  "services or find a helpline at findahelpline.com. This assistant can't provide crisis support.")
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|me|my|myself)\b", re.I)

_DIAGNOSE = re.compile(
    r"\b(do i have|diagnose (?:me|him|her|them|my)|what(?:'s| is) my diagnosis|am i (?:depressed|bipolar|ocd|ptsd)|"
    r"does (?:he|she|my \w+) have)\b",
    re.I,
)
NO_DIAGNOSIS_MESSAGE = (
    "I can't diagnose anyone. I can summarise what the PubMed literature says, and the overlap check can show "
    "which features tell similar conditions apart, so a clinician can ask about them."
)


@dataclass
class Screen:
    ok: bool
    kind: str = "ok"          # ok | crisis | risk_topic | diagnosis_request
    message: str = ""

    def to_dict(self) -> dict:
        out = {"ok": self.ok, "kind": self.kind, "message": self.message}
        if self.kind in ("crisis", "risk_topic"):
            out["resources"] = CRISIS_RESOURCES
        return out


def screen(text: str, topical_ok: bool = False) -> Screen:
    """topical_ok=True for literature questions: a non-first-person risk mention is allowed through."""
    if RISK.search(text or ""):
        if topical_ok and not _FIRST_PERSON.search(text):
            return Screen(True, "risk_topic", "If this is about you or someone at risk now: call or text 988 "
                                              "in the US, or local emergency services.")
        return Screen(False, "crisis", CRISIS_MESSAGE)
    if _DIAGNOSE.search(text or ""):
        return Screen(False, "diagnosis_request", NO_DIAGNOSIS_MESSAGE)
    return Screen(True)
