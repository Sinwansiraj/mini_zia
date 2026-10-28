"""
Ask Zia — optional grounded AI layer.

Turns the deterministic analysis results into a compact set of FACTS and sends
them to any OpenAI-compatible chat API (Groq, OpenAI, OpenRouter, Google's
OpenAI-compatible endpoint, …) so the model can rephrase or answer questions.

It is *grounded*: the system prompt forbids inventing or changing numbers, so
the LLM can only explain what the engine already computed. No third-party SDK
is required — requests are made with the standard library (urllib).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are Mini Zia, a concise business-insight assistant. You are given FACTS "
    "computed by a deterministic analytics engine. Answer ONLY using these facts. "
    "Never invent, estimate, or change any numbers; if the answer is not in the "
    "facts, say you don't have that information. Keep answers short, clear, and "
    "business-friendly."
)


def build_context(results: dict, cfg: dict) -> str:
    """Flatten the computed results into a compact factual context string."""
    insights = results.get("insights", {}) or {}
    metric = cfg.get("primary_metric", "metric")
    unit = cfg.get("currency") or "n/a"
    lines: List[str] = [
        f"Primary metric analysed: {metric} (unit: {unit})",
        f"Comparison: current {cfg.get('current_days', '?')}d vs previous {cfg.get('previous_days', '?')}d",
        f"Priority: {insights.get('priority', '?')}",
        f"Summary: {insights.get('summary', '')}",
    ]
    if insights.get("details"):
        lines.append("Key findings:")
        lines += [f"  - {' '.join(d.split())}" for d in insights["details"]]
    if insights.get("recommendations"):
        lines.append("Recommendations:")
        lines += [f"  - {r.replace('💡', '').strip()}" for r in insights["recommendations"]]

    anomalies = results.get("anomalies", []) or []
    if anomalies:
        lines.append("Anomalies:")
        for a in anomalies[:5]:
            lines.append(
                f"  - {a.get('group', a.get('region', '?'))}: {a.get('metric', '')} "
                f"{float(a.get('deviation_pct', 0)):+.1f}% ({a.get('anomaly_type', '')})"
            )

    comp = results.get("comparisons")
    col = f"{metric}_change_pct"
    if comp is not None and len(comp) > 0 and col in comp.columns:
        dims = cfg.get("dimensions", []) or []
        top = comp.assign(_a=comp[col].abs()).nlargest(min(6, len(comp)), "_a")
        lines.append(f"Top movers by {metric} change:")
        for _, row in top.iterrows():
            label = " / ".join(str(row[d]) for d in dims if d in row) or "overall"
            lines.append(f"  - {label}: {row[col]:+.1f}%")
    return "\n".join(lines)


def build_messages(context: str, history: List[Tuple[str, str]], question: str) -> List[dict]:
    """Assemble the chat messages: grounded system prompt + history + new question."""
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nFACTS:\n{context}"}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    if question:
        messages.append({"role": "user", "content": question})
    return messages


def chat_completion(
    messages: List[dict],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 30,
    max_tokens: int = 500,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the reply text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - trusted config URL
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"LLM API error {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001 - surface any transport error to the UI
        raise RuntimeError(f"Could not reach the LLM: {exc}") from exc
