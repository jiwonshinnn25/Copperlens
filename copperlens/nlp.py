"""Market-report NLP pipeline: PDF -> text -> structured trading-desk brief.

Two interchangeable engines share one output schema:
  * LLM engine (Anthropic API) - used when ANTHROPIC_API_KEY is set
  * Rule-based engine          - lexicon + regex, works offline / as fallback
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any

from pypdf import PdfReader

MODEL = os.getenv("CL_MODEL", "claude-sonnet-5-5")
MAX_CHARS = 24_000

SYSTEM_PROMPT = """You are a commodities research analyst supporting a trading desk.
Read the daily market report and return ONLY a JSON object (no markdown, no prose) with:
{
  "headline": "<=14 word headline",
  "summary": ["3-4 concise bullet strings"],
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "signals": [{"factor": "Supply|Demand|Inventory|Macro|Policy", "direction": "bullish|bearish|neutral", "detail": "<=20 words"}],
  "price_levels": ["any explicit support/resistance/target levels quoted, verbatim"]
}
Use only information in the report. If something is not stated, omit it."""


# ---------------------------------------------------------------- PDF
def extract_text(source: bytes | io.BytesIO | str) -> str:
    """Extract text from a PDF given raw bytes, a file-like object or a path."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    reader = PdfReader(source)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    return re.sub(r"[ \t]+", " ", text).strip()


# -------------------------------------------------------- rule engine
# (pattern, factor, direction, weight)
LEXICON: list[tuple[str, str, str, float]] = [
    (r"inventor\w* (?:draw|fell|declin|drop)|stocks? (?:fell|drew|declin|dropp)|drawdown", "Inventory", "bullish", 2),
    (r"inventor\w* (?:build|rose|rise|increas)|stocks? (?:rose|climb|build|increas)", "Inventory", "bearish", 2),
    (r"strike|outage|disruption|mine clos|force majeure|shortage|lower output|production cut", "Supply", "bullish", 2),
    (r"oversuppl|surplus|record output|production (?:rose|rise|increase)|ramp[- ]?up", "Supply", "bearish", 2),
    (r"demand (?:growth|strength|recover|surge)|restock|grid invest|data cent|ev demand|stimulus|pmi (?:expan|rose|above)", "Demand", "bullish", 2),
    (r"weak demand|demand (?:weak|slow|soft|fell|drop)|slowdown|pmi (?:contract|fell|below)|property slump", "Demand", "bearish", 2),
    (r"weaker dollar|dollar (?:weak|fell|slid)|rate cut|dovish", "Macro", "bullish", 1),
    (r"stronger dollar|dollar (?:strength|firm|rall|rose)|rate hike|hawkish|risk[- ]off", "Macro", "bearish", 1),
    (r"tariff|export (?:ban|restriction)|sanction", "Policy", "bearish", 1),
]
PRICE_RE = re.compile(r"(?:USD|US\$|\$)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:/t|/lb|/oz|per (?:tonne|ton|lb|ounce)))?", re.I)
LEVEL_RE = re.compile(r"support|resistance|target|breakout|key level", re.I)


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", flat) if len(s.strip()) > 25]


def summarise_rule_based(text: str) -> dict[str, Any]:
    sents = _sentences(text)
    scored, signals, score = [], [], 0.0
    for s in sents:
        hit_w = 0.0
        for pat, factor, direction, w in LEXICON:
            if re.search(pat, s, re.I):
                hit_w += w
                score += w if direction == "bullish" else -w
                signals.append({"factor": factor, "direction": direction, "detail": _clip(s, 140)})
        scored.append((hit_w + (0.5 if PRICE_RE.search(s) else 0), s))

    top = sorted(range(len(sents)), key=lambda i: -scored[i][0])[:4]
    bullets = [_clip(sents[i], 200) for i in sorted(top) if scored[i][0] > 0] or [_clip(s, 200) for s in sents[:3]]

    seen, uniq = set(), []
    for sig in signals:  # de-dup on evidence sentence + factor
        key = (sig["factor"], sig["detail"])
        if key not in seen:
            seen.add(key)
            uniq.append(sig)

    n_hits = max(len(uniq), 1)
    sentiment = "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral"
    confidence = round(min(0.9, 0.35 + 0.08 * abs(score) / max(1, n_hits) * n_hits ** 0.5), 2)
    levels = [_clip(s, 160) for s in sents if LEVEL_RE.search(s) and PRICE_RE.search(s)][:3]

    return {
        "headline": _clip(_first_line(text) or (sents[0] if sents else ""), 90) or "No readable text",
        "summary": bullets,
        "sentiment": sentiment,
        "confidence": confidence,
        "signals": uniq[:8],
        "price_levels": levels,
        "engine": "rule-based",
    }


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ----------------------------------------------------------- LLM engine
def _parse_json(raw: str) -> dict[str, Any]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("LLM response contained no JSON object")
    return json.loads(raw[start : end + 1])


def _normalise(d: dict[str, Any], engine: str) -> dict[str, Any]:
    sent = str(d.get("sentiment", "neutral")).lower()
    sent = sent if sent in {"bullish", "bearish", "neutral"} else "neutral"
    sigs = []
    for s in d.get("signals", [])[:8]:
        direction = str(s.get("direction", "neutral")).lower()
        sigs.append({
            "factor": str(s.get("factor", "Other")),
            "direction": direction if direction in {"bullish", "bearish", "neutral"} else "neutral",
            "detail": str(s.get("detail", "")),
        })
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "headline": str(d.get("headline", "")),
        "summary": [str(x) for x in d.get("summary", [])][:5],
        "sentiment": sent,
        "confidence": conf,
        "signals": sigs,
        "price_levels": [str(x) for x in d.get("price_levels", [])][:5],
        "engine": engine,
    }


def summarise_with_llm(text: str) -> dict[str, Any]:
    import anthropic  # imported lazily so the app runs without the SDK/key

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text[:MAX_CHARS]}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _normalise(_parse_json(raw), engine=f"LLM ({MODEL})")


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def summarise(text: str, use_llm: bool = True) -> dict[str, Any]:
    """LLM summary when possible; automatic graceful fallback to rules."""
    if not text.strip():
        return {**summarise_rule_based(""), "note": "No extractable text (scanned PDF?)"}
    if use_llm and llm_available():
        try:
            return summarise_with_llm(text)
        except Exception as exc:  # network/auth/parse errors -> fallback
            out = summarise_rule_based(text)
            out["note"] = f"LLM unavailable ({type(exc).__name__}); used rule-based engine."
            return out
    return summarise_rule_based(text)
