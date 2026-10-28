"""
Insight Engine for Mini Zia
Generates natural language insights from analytical results.

Metric-, dimension- and unit-agnostic: the primary metric, driver metrics and
grouping dimensions are passed in by name, so the engine works on any business
data, not just sales/leads/region.
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


def _label(name: str) -> str:
    """Turn a column name into a human label, e.g. 'conversion_rate' -> 'Conversion Rate'."""
    return str(name).replace('_', ' ').title()


def _fmt(value: float, currency: str = '') -> str:
    """Format a numeric value with an optional unit/currency prefix."""
    try:
        return f"{currency}{value:,.0f}"
    except (TypeError, ValueError):
        return f"{currency}{value}"


class InsightEngine:
    """
    Converts analytical findings into human-readable business insights.
    """

    def __init__(self):
        self.insights = []
        self.priority_map = {'critical': 1, 'high': 2, 'medium': 3, 'low': 4}

    def generate_period_comparison_insight(self, comparison_data: Dict,
                                           primary_metric: str,
                                           currency: str = '',
                                           location: str = '') -> str:
        """Generate insight for a period-over-period comparison of the primary metric."""
        label = _label(primary_metric)
        change_pct = comparison_data.get(f'{primary_metric}_change_pct', 0)
        current_value = comparison_data.get(f'{primary_metric}_current', 0)

        if abs(change_pct) > 25:
            emoji, priority = "🚨", "critical"
        elif abs(change_pct) > 15:
            emoji, priority = "⚠️", "high"
        else:
            emoji, priority = "📊", "medium"

        if change_pct > 0:
            trend, direction = "increased", "↗️"
        elif change_pct < 0:
            trend, direction = "dropped", "↘️"
        else:
            trend, direction = "remained stable", "→"

        return (
            f"{emoji} {label} {trend} by {abs(change_pct):.1f}%{location}\n"
            f"   Current: {_fmt(current_value, currency)} {direction}\n"
            f"   Priority: {priority.upper()}"
        )

    def generate_root_cause_insight(self, cause_data: Dict, location: str = '') -> str:
        """Generate insight explaining the likely root-cause driver(s)."""
        insights = []
        for driver in cause_data.get('potential_drivers', []):
            if driver.get('likely_cause'):
                driver_name = _label(driver['driver'])
                change = driver['change_pct']
                impact = "declined" if change < 0 else "increased"
                insights.append(
                    f"🔍 Primary cause{location}: {driver_name} {impact} by {abs(change):.1f}%"
                )
        return "\n".join(insights) if insights else "🔍 No clear root cause identified"

    def generate_anomaly_alert(self, anomaly: Dict) -> str:
        """Generate an alert for a detected anomaly (dimension-agnostic)."""
        group = anomaly.get('group', anomaly.get('region', 'Unknown'))
        metric = _label(anomaly.get('metric', ''))
        deviation = anomaly.get('deviation_pct', 0)
        anomaly_type = anomaly.get('anomaly_type', '')

        if 'below' in anomaly_type:
            emoji, comparison = "📉", "below"
        else:
            emoji, comparison = "📈", "above"

        return (
            f"{emoji} Anomaly: {group}\n"
            f"   {metric} is {abs(deviation):.1f}% {comparison} the average\n"
            f"   Worth investigating"
        )

    def generate_comprehensive_insight(self,
                                       comparison: pd.DataFrame,
                                       cause_analysis: Optional[Dict],
                                       anomalies: List[Dict],
                                       primary_metric: str = 'sales',
                                       dimensions: Optional[List[str]] = None,
                                       currency: str = '',
                                       filters: Optional[Dict[str, str]] = None) -> Dict:
        """
        Generate a comprehensive insight combining all analyses.

        Args:
            comparison: period-comparison DataFrame (columns named <metric>_change_pct etc.)
            cause_analysis: root-cause dict from TrendAnalyzer.analyze_cause_effect
            anomalies: list of anomaly dicts
            primary_metric: the main metric column name being analysed
            dimensions: grouping dimension column names
            currency: optional unit/currency symbol for display
            filters: optional {dimension: value} filters applied
        """
        label = _label(primary_metric)
        filters = filters or {}
        change_col = f'{primary_metric}_change_pct'

        insight_package = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': "",
            'details': [],
            'recommendations': [],
            'priority': 'medium',
        }

        # Build a location string from the applied filters, e.g. " in Chennai (Enterprise)".
        location = ""
        filter_vals = [str(v) for v in filters.values() if v]
        if filter_vals:
            location = " in " + " / ".join(filter_vals)

        comp_data = comparison.copy()
        if change_col not in comp_data.columns or len(comp_data) == 0:
            insight_package['summary'] = "No significant changes detected"
            return insight_package

        comp_data['_abs_change'] = comp_data[change_col].abs()
        main_row = comp_data.nlargest(1, '_abs_change').iloc[0]
        main_change = main_row[change_col]

        if abs(main_change) > 25:
            insight_package['priority'] = 'critical'
        elif abs(main_change) > 15:
            insight_package['priority'] = 'high'

        trend = "increased" if main_change > 0 else "decreased"
        insight_package['summary'] = (
            f"{label} {trend} by {abs(main_change):.1f}% compared to the previous period{location}"
        )

        # Per-row location (the specific group driving the headline change)
        row_vals = []
        for dim in (dimensions or []):
            if dim in main_row and pd.notna(main_row[dim]):
                row_vals.append(str(main_row[dim]))
        row_location = (" in " + " / ".join(row_vals)) if row_vals else location

        insight_package['details'].append(
            self.generate_period_comparison_insight(
                main_row.to_dict(), primary_metric, currency=currency, location=row_location
            )
        )

        if cause_analysis:
            cause_insight = self.generate_root_cause_insight(cause_analysis, location=row_location)
            if cause_insight:
                insight_package['details'].append(cause_insight)

        for anomaly in anomalies[:3]:
            insight_package['details'].append(self.generate_anomaly_alert(anomaly))

        insight_package['recommendations'] = self._generate_recommendations(
            main_change, cause_analysis, label, row_location
        )
        return insight_package

    def _generate_recommendations(self, change_pct: float,
                                  cause_analysis: Optional[Dict],
                                  metric_label: str,
                                  location: str = '') -> List[str]:
        """Generate generic, metric-agnostic recommendations."""
        recommendations = []
        drivers = (cause_analysis or {}).get('potential_drivers', [])
        likely = [d for d in drivers if d.get('likely_cause')]

        if change_pct < -15:
            recommendations.append(
                f"💡 {metric_label} fell sharply{location} — investigate and act before it compounds."
            )
            for d in likely:
                recommendations.append(
                    f"💡 Focus on {_label(d['driver'])} (changed {d['change_pct']:+.1f}%), the likely driver."
                )
        elif change_pct > 15:
            recommendations.append(
                f"💡 {metric_label} rose strongly{location} — identify what worked and replicate it elsewhere."
            )
            for d in likely:
                recommendations.append(
                    f"💡 Sustain the gain in {_label(d['driver'])} ({d['change_pct']:+.1f}%) that drove it."
                )
        return recommendations

    def format_insight_report(self, insight_package: Dict) -> str:
        """Format the insight package into a readable text report."""
        priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
        emoji = priority_emoji.get(insight_package['priority'], '🟡')

        report = "\n" + "=" * 60 + "\n"
        report += f"{emoji} BUSINESS INSIGHT REPORT\n"
        report += "=" * 60 + "\n\n"
        report += f"📅 Generated: {insight_package['timestamp']}\n"
        report += f"🎯 Priority: {insight_package['priority'].upper()}\n\n"
        report += "📊 SUMMARY\n" + "-" * 60 + "\n"
        report += insight_package['summary'] + "\n\n"

        if insight_package['details']:
            report += "📋 DETAILS\n" + "-" * 60 + "\n"
            for detail in insight_package['details']:
                report += detail + "\n\n"

        if insight_package['recommendations']:
            report += "💡 RECOMMENDATIONS\n" + "-" * 60 + "\n"
            for rec in insight_package['recommendations']:
                report += rec + "\n"

        report += "\n" + "=" * 60 + "\n"
        return report
