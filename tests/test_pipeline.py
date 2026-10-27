"""
Tests for the Mini Zia insight pipeline.

These cover the engine only (data generation -> anomaly detection -> period
comparison -> insights); the Streamlit UI is not imported here.
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

from data_generator import generate_business_data, get_previous_period, get_recent_period
from main import MiniZia

REQUIRED_COLUMNS = {"date", "region", "segment", "sales", "leads", "conversion_rate"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_business_data(days=90, seed=42)


@pytest.fixture(scope="module")
def results(df: pd.DataFrame) -> dict:
    zia = MiniZia(enable_llm=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return zia.analyze(df, current_period_days=7, previous_period_days=7)


# ── Data generation ──────────────────────────────────────────────────────────


def test_sample_data_has_required_columns(df: pd.DataFrame):
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_sample_data_row_count(df: pd.DataFrame):
    # 90 days * 5 regions * 2 segments
    assert len(df) == 90 * 5 * 2


def test_date_column_is_datetime(df: pd.DataFrame):
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_injected_anomaly_targets_exist(df: pd.DataFrame):
    assert "Chennai" in df["region"].unique()
    assert "Enterprise" in df["segment"].unique()


# ── Period extraction ────────────────────────────────────────────────────────


def test_recent_and_previous_periods_are_non_empty(df: pd.DataFrame):
    current = get_recent_period(df, days=7)
    previous = get_previous_period(df, days=7, offset=7)
    assert len(current) > 0
    assert len(previous) > 0


def test_periods_do_not_overlap(df: pd.DataFrame):
    current = get_recent_period(df, days=7)
    previous = get_previous_period(df, days=7, offset=7)
    assert previous["date"].max() < current["date"].min()


# ── Pipeline ─────────────────────────────────────────────────────────────────


def test_analyze_returns_expected_keys(results: dict):
    for key in ("anomalies", "comparisons", "root_causes", "insights"):
        assert key in results


def test_insight_package_structure(results: dict):
    insights = results["insights"]
    assert set(insights).issuperset({"summary", "details", "recommendations", "priority"})
    assert isinstance(insights["summary"], str) and insights["summary"]
    assert insights["priority"] in VALID_PRIORITIES


def test_anomalies_detected(results: dict):
    # The generator injects deliberate anomalies, so the recent window should flag some.
    assert len(results["anomalies"]) >= 1
    sample = results["anomalies"][0]
    assert {"region", "metric", "deviation_pct", "anomaly_type"}.issubset(sample)


def test_comparison_has_change_columns(results: dict):
    comparison = results["comparisons"]
    assert len(comparison) > 0
    assert "sales_change_pct" in comparison.columns


def test_summary_metrics(df: pd.DataFrame, results: dict):
    zia = MiniZia(enable_llm=False)
    metrics = zia.generate_summary_metrics(results)
    assert isinstance(metrics["total_anomalies"], int)
    assert metrics["priority"] in VALID_PRIORITIES


def test_region_filter_runs(df: pd.DataFrame):
    zia = MiniZia(enable_llm=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res = zia.analyze(df, current_period_days=7, previous_period_days=7, region_filter="Chennai")
    assert res["insights"]["priority"] in VALID_PRIORITIES
