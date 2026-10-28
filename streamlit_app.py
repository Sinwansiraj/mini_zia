"""
Mini Zia — Automated Business Insight Generator (web demo).

A thin Streamlit UI over the rule-based insight engine. It runs the full
pipeline (anomaly detection -> period comparison -> root cause -> natural
language insights) on synthetic KPI data OR on any business CSV you upload and
map, and shows the results the way a business user would read them.

The insights are fully deterministic and explainable. An optional LLM
rephrasing layer exists in the codebase but is intentionally disabled in this
public demo (no API keys, no cost, nothing sent to a third party).
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd
import streamlit as st

from data_generator import generate_business_data
from insight_engine import InsightEngine
from main import MiniZia

PRIORITY_STYLE = {
    "critical": ("🔴", "#ef4444"),
    "high": ("🟠", "#f59e0b"),
    "medium": ("🟡", "#eab308"),
    "low": ("🟢", "#22c55e"),
}

st.set_page_config(page_title="Mini Zia — Business Insight Generator", page_icon="🤖", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample_data(days: int, seed: int = 42) -> pd.DataFrame:
    return generate_business_data(days=days, seed=seed)


def run_pipeline(df: pd.DataFrame, cfg: dict) -> tuple[dict, str]:
    """Run MiniZia.analyze while capturing its console output. Returns (results, log)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        zia = MiniZia(
            anomaly_threshold_std=cfg["threshold_std"],
            anomaly_threshold_pct=cfg["threshold_pct"],
            significance_threshold=cfg["significance"],
            enable_llm=False,
        )
        results = zia.analyze(
            df,
            current_period_days=cfg["current_days"],
            previous_period_days=cfg["previous_days"],
            primary_metric=cfg["primary_metric"],
            driver_metrics=cfg["driver_metrics"],
            dimensions=cfg["dimensions"],
            mean_metrics=cfg["mean_metrics"],
            currency=cfg["currency"],
            filters=cfg["filters"],
        )
        results["_summary_metrics"] = zia.generate_summary_metrics(results)
        results["_report"] = InsightEngine().format_insight_report(results["insights"])
    return results, buffer.getvalue()


# ── Header ───────────────────────────────────────────────────────────────────

st.title("🤖 Mini Zia")
st.caption("Automated Business Insight Generator — turns any KPI data into plain-English insights.")

st.info(
    "**Demo note:** insights come from a **deterministic, rule-based engine** — fully explainable, no "
    "black box. Use the built-in **sample data**, or **upload your own CSV** and map the columns. An "
    "optional LLM layer exists in the code but is **disabled here** (no keys, nothing sent anywhere).",
    icon="ℹ️",
)

# ── Sidebar: data source + column mapping ────────────────────────────────────

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Sample data", "Upload CSV"], index=0)

cfg: dict = {}
working_df: pd.DataFrame | None = None

if source == "Sample data":
    with st.sidebar:
        days = st.slider("Days of history to generate", 30, 120, 90, step=10)
    working_df = load_sample_data(days)
    cfg.update(
        primary_metric="sales",
        driver_metrics=["leads", "conversion_rate"],
        dimensions=["region", "segment"],
        mean_metrics=["conversion_rate"],
        currency="₹",
    )
else:
    with st.sidebar:
        st.download_button(
            "⬇ Download sample CSV template",
            data=load_sample_data(30).to_csv(index=False).encode("utf-8"),
            file_name="mini_zia_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
        upload = st.file_uploader("Upload a KPI CSV", type=["csv"])
    if upload is None:
        st.warning("⬅ Upload a CSV (or switch to **Sample data**) to begin.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    if len(raw) == 0:
        st.error("The uploaded file has no rows.")
        st.stop()

    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    all_cols = raw.columns.tolist()
    cat_cols = [c for c in all_cols if c not in numeric_cols]

    if not numeric_cols:
        st.error("No numeric columns found — Mini Zia needs at least one numeric metric to analyse.")
        st.stop()

    with st.sidebar:
        st.header("Map your columns")
        date_guess = next((c for c in all_cols if "date" in c.lower() or "time" in c.lower()), all_cols[0])
        date_col = st.selectbox("Date column", all_cols, index=all_cols.index(date_guess))
        primary_metric = st.selectbox("Metric to analyse", numeric_cols, index=0)
        driver_metrics = st.multiselect(
            "Driver metrics (optional)", [c for c in numeric_cols if c != primary_metric]
        )
        dimensions = st.multiselect(
            "Dimensions to break down by", [c for c in cat_cols if c != date_col],
            default=[c for c in cat_cols if c != date_col][:2],
        )
        mean_metrics = st.multiselect(
            "Which metrics are rates/% (averaged, not summed)?",
            [primary_metric, *driver_metrics],
            help="e.g. conversion rate, CTR — these are averaged across groups instead of summed.",
        )
        currency = st.text_input("Unit / currency symbol (optional)", value="")

    # Canonicalise: rename the chosen date column to 'date' and parse it.
    working_df = raw.rename(columns={date_col: "date"}).copy()
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
    if working_df["date"].isna().all():
        st.error(f"Could not parse dates in '{date_col}'. Make sure it holds real dates.")
        st.stop()
    working_df = working_df.dropna(subset=["date"])
    if len(working_df) > 200_000:
        st.error("Please keep the dataset under 200,000 rows for this demo.")
        st.stop()

    cfg.update(
        primary_metric=primary_metric,
        driver_metrics=driver_metrics,
        dimensions=dimensions,
        mean_metrics=mean_metrics,
        currency=currency.strip(),
    )

# ── Sidebar: window, filters, thresholds, run ────────────────────────────────

with st.sidebar:
    st.header("Comparison window")
    cfg["current_days"] = st.slider("Current period (days)", 1, 30, 7)
    cfg["previous_days"] = st.slider("Previous period (days)", 1, 30, 7)

    st.header("Filters")
    filters: dict[str, str] = {}
    for dim in cfg["dimensions"]:
        values = ["All", *[str(v) for v in sorted(working_df[dim].dropna().unique())]]
        choice = st.selectbox(dim, values, index=0, key=f"filter_{dim}")
        if choice != "All":
            filters[dim] = choice
    cfg["filters"] = filters

    with st.expander("Advanced thresholds"):
        cfg["threshold_std"] = st.number_input("Anomaly threshold (std devs)", 0.5, 5.0, 2.0, 0.5)
        cfg["threshold_pct"] = st.number_input("Anomaly threshold (% change)", 5.0, 50.0, 15.0, 5.0)
        cfg["significance"] = st.number_input("Trend significance (%)", 5.0, 50.0, 10.0, 5.0)

    run = st.button("▶ Run analysis", type="primary", use_container_width=True)

# ── Intro / run ──────────────────────────────────────────────────────────────

if not run:
    st.subheader("How to use this")
    st.markdown(
        "1. Use **Sample data**, or **Upload CSV** and map your columns (date, the metric to analyse, "
        "optional drivers, and dimensions to break down by).\n"
        "2. Choose the **comparison window** (e.g. this week vs last week) and any **filters**.\n"
        "3. Click **Run analysis** — Mini Zia flags anomalies, compares periods, finds the likely cause, "
        "and writes an executive summary with recommendations."
    )
    st.dataframe(working_df.head(12), use_container_width=True)
    st.stop()

try:
    with st.spinner("Analysing…"):
        results, log = run_pipeline(working_df, cfg)
except Exception as exc:  # noqa: BLE001
    st.error(f"Analysis failed: {exc}")
    st.stop()

insights = results["insights"]
metrics = results["_summary_metrics"]
priority = insights.get("priority", "medium")
emoji, color = PRIORITY_STYLE.get(priority, ("🟡", "#eab308"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records analysed", f"{len(working_df):,}")
c2.metric("Groups compared", metrics.get("regions_analyzed", "—"))
c3.metric("Anomalies detected", metrics.get("total_anomalies", 0))
c4.metric("Significant changes", metrics.get("significant_changes", 0))

st.markdown(
    f"<div style='padding:14px 18px;border-radius:12px;border:1px solid {color}55;"
    f"background:{color}12;margin:8px 0 4px'>"
    f"<span style='font-weight:700'>{emoji} Priority: {priority.upper()}</span><br>"
    f"<span style='font-size:1.1rem'>{insights.get('summary', 'No significant changes detected')}</span></div>",
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Key insights")
    details = insights.get("details", [])
    if details:
        for d in details:
            lines = d.strip().splitlines()
            st.markdown(f"- {lines[0]}")
            if lines[1:]:
                st.caption("  \n".join(line.strip() for line in lines[1:]))
    else:
        st.caption("No notable details for this selection.")
with right:
    st.subheader("Recommendations")
    recs = insights.get("recommendations", [])
    if recs:
        for r in recs:
            st.markdown(f"- {r.replace('💡', '').strip()}")
    else:
        st.caption("No specific recommendations — metrics look stable.")

anomalies = results.get("anomalies", [])
if anomalies:
    st.subheader("Anomalies")
    acols = st.columns(min(3, len(anomalies)))
    for i, a in enumerate(anomalies[:3]):
        with acols[i % len(acols)]:
            dev = float(a.get("deviation_pct", 0))
            arrow = "📉" if dev < 0 else "📈"
            label = str(a.get("group", a.get("region", "—")))
            st.metric(
                f"{arrow} {label} · {str(a.get('metric', '')).replace('_', ' ').title()}",
                f"{dev:+.1f}%",
                help=str(a.get("anomaly_type", "")).replace("_", " "),
            )

# Trend chart — first dimension by the primary metric over time
st.subheader(f"{cfg['primary_metric'].replace('_', ' ').title()} trend")
try:
    dims = cfg["dimensions"]
    if dims:
        pivot = working_df.groupby(["date", dims[0]])[cfg["primary_metric"]].sum().unstack()
    else:
        pivot = working_df.groupby("date")[cfg["primary_metric"]].sum()
    st.line_chart(pivot, height=280)
except Exception:  # noqa: BLE001
    st.caption("Trend chart unavailable for this data shape.")

comparison = results.get("comparisons")
if comparison is not None and len(comparison) > 0:
    with st.expander("Period-over-period comparison (table)"):
        metric_cols = [cfg["primary_metric"], *cfg["driver_metrics"]]
        show_cols = cfg["dimensions"] + [f"{m}_change_pct" for m in metric_cols if f"{m}_change_pct" in comparison.columns]
        show_cols = [c for c in show_cols if c in comparison.columns]
        st.dataframe(comparison[show_cols].round(2) if show_cols else comparison.round(2), use_container_width=True)

with st.expander("Full report & engine trace"):
    st.text(results.get("_report", ""))
    if log.strip():
        st.caption("Pipeline log")
        st.code(log, language="text")

st.divider()
st.caption(
    "Mini Zia · rule-based (explainable) insight engine · works on any business CSV · "
    "optional LLM rephrasing disabled in this demo."
)
