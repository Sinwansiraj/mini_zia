"""
Mini Zia - Automated Business Insight Generator
Main orchestrator that brings all components together.

The analysis is metric-, dimension- and unit-agnostic: pass the primary metric,
driver metrics and grouping dimensions by name to analyse any business data.
The defaults reproduce the built-in sales/leads/region sample.
"""

import pandas as pd
from datetime import datetime
import sys

# Make Unicode/emoji console output safe on every platform (e.g. Windows cp1252
# consoles would otherwise raise UnicodeEncodeError). No-op where already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Import all modules
from data_generator import (
    generate_business_data,
    get_recent_period,
    get_previous_period,
)
from anomaly_detector import AnomalyDetector, format_anomaly_report
from trend_analyzer import TrendAnalyzer
from insight_engine import InsightEngine
from llm_rephraser import LLMRephraser


class MiniZia:
    """Main orchestrator for the automated business insight generator."""

    def __init__(self,
                 anomaly_threshold_std=2.0,
                 anomaly_threshold_pct=15.0,
                 significance_threshold=10.0,
                 enable_llm=False,
                 llm_api_key=None):
        self.anomaly_detector = AnomalyDetector(
            threshold_std=anomaly_threshold_std,
            threshold_pct=anomaly_threshold_pct,
        )
        self.trend_analyzer = TrendAnalyzer(significance_threshold=significance_threshold)
        self.insight_engine = InsightEngine()
        self.llm_rephraser = LLMRephraser(api_key=llm_api_key if enable_llm else None)
        self.enable_llm = enable_llm

    def analyze(self,
                df: pd.DataFrame,
                current_period_days: int = 7,
                previous_period_days: int = 7,
                primary_metric: str = 'sales',
                driver_metrics=None,
                dimensions=None,
                mean_metrics=None,
                currency: str = '',
                filters=None) -> dict:
        """
        Run the complete analysis pipeline on any business data.

        Args:
            df: input data (must include a 'date' column)
            current_period_days / previous_period_days: comparison windows
            primary_metric: the metric column to analyse (e.g. 'sales', 'revenue')
            driver_metrics: candidate driver columns (default ['leads', 'conversion_rate'])
            dimensions: grouping columns (default ['region', 'segment'])
            mean_metrics: metrics aggregated by mean/rate instead of sum (default ['conversion_rate'])
            currency: optional unit/currency symbol for display
            filters: optional {dimension: value} filters
        """
        if driver_metrics is None:
            driver_metrics = ['leads', 'conversion_rate']
        if dimensions is None:
            dimensions = ['region', 'segment']
        if mean_metrics is None:
            mean_metrics = ['conversion_rate']
        filters = filters or {}
        drivers = [m for m in driver_metrics if m != primary_metric]
        metrics = [primary_metric] + drivers

        print("\n🚀 Starting Mini Zia Analysis...")
        print("=" * 60)

        current = get_recent_period(df, days=current_period_days)
        previous = get_previous_period(df, days=previous_period_days, offset=current_period_days)

        # Apply dimension filters
        for dim, val in filters.items():
            if val and dim in current.columns:
                current = current[current[dim] == val]
                previous = previous[previous[dim] == val]

        # Dimensions still free to group by (filtered ones are now constant)
        group_dims = [d for d in dimensions if not filters.get(d)]
        anomaly_dim = group_dims[0] if group_dims else (dimensions[0] if dimensions else None)

        results = {
            'timestamp': datetime.now(),
            'filters': filters,
            'primary_metric': primary_metric,
            'driver_metrics': drivers,
            'dimensions': dimensions,
            'currency': currency,
            'anomalies': [],
            'comparisons': None,
            'root_causes': None,
            'insights': None,
        }

        # Step 1: anomalies across the primary dimension
        if anomaly_dim:
            results['anomalies'] = self.anomaly_detector.find_regional_anomalies(
                current, metric=primary_metric,
                period_days=current_period_days, dimension=anomaly_dim,
            )

        # Step 2: period-over-period comparison
        comparison = self.anomaly_detector.detect_significant_changes(
            current, previous, metrics=metrics,
            group_by=group_dims if group_dims else None,
            mean_metrics=mean_metrics,
        )
        results['comparisons'] = comparison

        # Step 3: root cause on the biggest mover
        change_col = f'{primary_metric}_change_pct'
        if len(comparison) > 0 and change_col in comparison.columns:
            tmp = comparison.assign(_abs=comparison[change_col].abs())
            top = tmp.nlargest(1, '_abs').iloc[0]
            rca_current, rca_previous = current, previous
            for dim in group_dims:
                if dim in top and pd.notna(top[dim]):
                    rca_current = rca_current[rca_current[dim] == top[dim]]
                    rca_previous = rca_previous[rca_previous[dim] == top[dim]]
            if drivers:
                results['root_causes'] = self.trend_analyzer.analyze_cause_effect(
                    rca_current, rca_previous, target_metric=primary_metric,
                    driver_metrics=drivers, mean_metrics=mean_metrics,
                )

        # Step 4: natural-language insights
        insight_package = self.insight_engine.generate_comprehensive_insight(
            comparison=comparison,
            cause_analysis=results.get('root_causes'),
            anomalies=results['anomalies'],
            primary_metric=primary_metric,
            dimensions=group_dims,
            currency=currency,
            filters=filters,
        )
        results['insights'] = insight_package
        print(self.insight_engine.format_insight_report(insight_package))

        # Step 5: optional LLM enhancement
        if self.enable_llm:
            enhanced = self.llm_rephraser.enhance_insight(
                insight_package['summary'], tone='professional'
            )
            print(f"Enhanced Summary: {enhanced}")

        return results

    def generate_summary_metrics(self, results: dict) -> dict:
        """Generate top-line summary metrics from analysis results."""
        summary = {
            'total_anomalies': len(results.get('anomalies', [])),
            'critical_issues': 0,
            'priority': results['insights']['priority'] if results.get('insights') else 'medium',
        }
        comparison = results.get('comparisons')
        if comparison is not None:
            summary['regions_analyzed'] = len(comparison)
            pm = results.get('primary_metric', 'sales')
            flag_col = f'{pm}_is_anomaly'
            summary['significant_changes'] = (
                int(comparison[flag_col].sum()) if flag_col in comparison.columns else 0
            )
        return summary


def main():
    """Demonstrate the complete workflow on the built-in sample data."""
    print("\n" + "=" * 60)
    print("🤖 MINI ZIA - Automated Business Insight Generator")
    print("=" * 60)

    df = generate_business_data(days=90)
    print(f"\n✓ Generated {len(df)} records across {df['date'].nunique()} days")

    zia = MiniZia(anomaly_threshold_std=2.0, anomaly_threshold_pct=15.0,
                  significance_threshold=10.0, enable_llm=False)

    print("\n" + "=" * 60 + "\n🎯 SCENARIO 1: Overall Business Analysis\n" + "=" * 60)
    results_overall = zia.analyze(df, current_period_days=7, previous_period_days=7, currency='₹')

    print("\n\n" + "=" * 60 + "\n🎯 SCENARIO 2: Chennai Region Deep Dive\n" + "=" * 60)
    results_chennai = zia.analyze(df, current_period_days=7, previous_period_days=7,
                                  currency='₹', filters={'region': 'Chennai'})

    print("\n\n" + "=" * 60 + "\n🎯 SCENARIO 3: Enterprise Segment Analysis\n" + "=" * 60)
    results_enterprise = zia.analyze(df, current_period_days=7, previous_period_days=7,
                                     currency='₹', filters={'segment': 'Enterprise'})

    print("\n\n" + "=" * 60 + "\n📈 EXECUTIVE SUMMARY\n" + "=" * 60)
    summary = zia.generate_summary_metrics(results_overall)
    print(f"\n✓ Compared {summary.get('regions_analyzed', 0)} groups")
    print(f"✓ Detected {summary['total_anomalies']} anomalies")
    print(f"✓ Found {summary.get('significant_changes', 0)} significant changes")
    print(f"✓ Overall Priority: {summary['priority'].upper()}")

    print("\n" + "=" * 60 + "\n✅ ANALYSIS COMPLETE\n" + "=" * 60)
    return results_overall, results_chennai, results_enterprise


if __name__ == "__main__":
    try:
        main()
        print("\n💡 TIP: analyze() accepts any primary_metric / driver_metrics / dimensions,")
        print("    so Mini Zia works on any business CSV — see streamlit_app.py.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
