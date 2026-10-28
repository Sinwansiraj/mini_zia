"""Tests for the Ask Zia grounding helpers (no network / API calls)."""

from __future__ import annotations

import pandas as pd

from ai_assistant import build_context, build_messages


def _results() -> dict:
    return {
        "insights": {
            "priority": "high",
            "summary": "Revenue dropped by 20% compared to the previous period",
            "details": ["⚠️ Revenue dropped by 20% in Gadget"],
            "recommendations": ["💡 Investigate Gadget before it compounds."],
        },
        "anomalies": [
            {"group": "Gadget", "metric": "revenue", "deviation_pct": -20.0, "anomaly_type": "below_average"}
        ],
        "comparisons": pd.DataFrame(
            {"product": ["Gadget", "Widget"], "revenue_change_pct": [-20.0, 5.0]}
        ),
        "_summary_metrics": {},
    }


def _cfg() -> dict:
    return {
        "primary_metric": "revenue",
        "currency": "$",
        "dimensions": ["product"],
        "current_days": 7,
        "previous_days": 7,
    }


def test_build_context_includes_the_computed_facts():
    ctx = build_context(_results(), _cfg())
    assert "revenue" in ctx
    assert "Revenue dropped by 20%" in ctx
    assert "Gadget" in ctx
    assert "Top movers" in ctx


def test_build_messages_is_grounded():
    ctx = build_context(_results(), _cfg())
    msgs = build_messages(ctx, [], "Which product should I worry about?")
    assert msgs[0]["role"] == "system"
    assert "FACTS" in msgs[0]["content"]
    assert "Never invent" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "Which product should I worry about?"}


def test_build_messages_includes_history():
    msgs = build_messages("facts", [("user", "hi"), ("assistant", "hello")], "next")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
