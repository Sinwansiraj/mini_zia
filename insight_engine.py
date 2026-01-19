"""
Insight Engine for Mini Zia
Generates natural language insights from analytical results.
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime


class InsightEngine:
    """
    Converts analytical findings into human-readable business insights.
    """
    
    def __init__(self):
        """Initialize the insight engine with templates."""
        self.insights = []
        self.priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
    
    def generate_period_comparison_insight(self, comparison_data: Dict,
                                          region: str = None,
                                          segment: str = None) -> str:
        """
        Generate insight for period-over-period comparison.
        
        Args:
            comparison_data (Dict): Comparison results
            region (str): Region name
            segment (str): Segment name
        
        Returns:
            str: Natural language insight
        """
        change_pct = comparison_data.get('sales_change_pct', 0)
        current_value = comparison_data.get('sales_current', 0)
        
        # Determine severity
        if abs(change_pct) > 25:
            emoji = "🚨"
            priority = "critical"
        elif abs(change_pct) > 15:
            emoji = "⚠️"
            priority = "high"
        else:
            emoji = "📊"
            priority = "medium"
        
        # Build location context
        location = ""
        if region and segment:
            location = f" in {region} ({segment})"
        elif region:
            location = f" in {region}"
        elif segment:
            location = f" for {segment}"
        
        # Determine trend word
        if change_pct > 0:
            trend = "increased"
            direction = "↗️"
        elif change_pct < 0:
            trend = "dropped"
            direction = "↘️"
        else:
            trend = "remained stable"
            direction = "→"
        
        insight = (
            f"{emoji} Sales {trend} by {abs(change_pct):.1f}%{location}\n"
            f"   Current: ₹{current_value:,.0f} {direction}\n"
            f"   Priority: {priority.upper()}"
        )
        
        return insight
    
    def generate_root_cause_insight(self, cause_data: Dict,
                                   region: str = None,
                                   segment: str = None) -> str:
        """
        Generate insight explaining root causes.
        
        Args:
            cause_data (Dict): Cause analysis results
            region (str): Region name
            segment (str): Segment name
        
        Returns:
            str: Root cause explanation
        """
        location = ""
        if region and segment:
            location = f" in {region} ({segment})"
        elif region:
            location = f" in {region}"
        
        insights = []
        
        for driver in cause_data.get('potential_drivers', []):
            if driver.get('likely_cause'):
                driver_name = driver['driver'].replace('_', ' ').title()
                change = driver['change_pct']
                
                if change < 0:
                    impact = "declined"
                else:
                    impact = "increased"
                
                insight = (
                    f"🔍 Primary cause{location}: {driver_name} {impact} by {abs(change):.1f}%"
                )
                insights.append(insight)
        
        return "\n".join(insights) if insights else "🔍 No clear root cause identified"
    
    def generate_segment_insight(self, segment_analysis: Dict,
                                region: str = None) -> str:
        """
        Generate insight about segment performance.
        
        Args:
            segment_analysis (Dict): Segment breakdown
            region (str): Region name
        
        Returns:
            str: Segment insight
        """
        location = f" in {region}" if region else ""
        
        segments = segment_analysis.get('segments', {})
        
        if not segments:
            return ""
        
        # Find the segment with largest absolute change
        max_change = None
        max_segment = None
        
        for segment_name, data in segments.items():
            if data.get('is_significant'):
                change = abs(data.get('change_pct', 0))
                if max_change is None or change > max_change:
                    max_change = change
                    max_segment = segment_name
        
        if max_segment:
            data = segments[max_segment]
            change = data.get('change_pct', 0)
            
            if change < 0:
                impact = "underperformed"
            else:
                impact = "outperformed"
            
            insight = (
                f"🎯 {max_segment} segment {impact}{location} "
                f"with {abs(change):.1f}% change"
            )
            return insight
        
        return ""
    
    def generate_anomaly_alert(self, anomaly: Dict) -> str:
        """
        Generate alert for detected anomalies.
        
        Args:
            anomaly (Dict): Anomaly details
        
        Returns:
            str: Alert message
        """
        region = anomaly.get('region', 'Unknown')
        metric = anomaly.get('metric', '').replace('_', ' ').title()
        deviation = anomaly.get('deviation_pct', 0)
        anomaly_type = anomaly.get('anomaly_type', '')
        
        if 'below' in anomaly_type:
            emoji = "📉"
            comparison = "below"
        else:
            emoji = "📈"
            comparison = "above"
        
        alert = (
            f"{emoji} Anomaly Alert: {region}\n"
            f"   {metric} is {abs(deviation):.1f}% {comparison} average\n"
            f"   Requires investigation"
        )
        
        return alert
    
    def generate_comprehensive_insight(self, 
                                      comparison: pd.DataFrame,
                                      cause_analysis: Dict,
                                      anomalies: List[Dict],
                                      region: str = None,
                                      segment: str = None) -> Dict:
        """
        Generate comprehensive insight combining all analyses.
        
        Args:
            comparison (pd.DataFrame): Period comparison data
            cause_analysis (Dict): Root cause analysis
            anomalies (List[Dict]): Detected anomalies
            region (str): Region filter
            segment (str): Segment filter
        
        Returns:
            Dict: Structured insight package
        """
        insight_package = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': "",
            'details': [],
            'recommendations': [],
            'priority': 'medium'
        }
        
        # Filter comparison data - only filter if columns exist
        comp_data = comparison.copy()
        if region and 'region' in comp_data.columns:
            comp_data = comp_data[comp_data['region'] == region]
        if segment and 'segment' in comp_data.columns:
            comp_data = comp_data[comp_data['segment'] == segment]
        
        if len(comp_data) == 0:
            insight_package['summary'] = "No significant changes detected"
            return insight_package
        
        # Get the row with largest absolute sales change
        comp_data['abs_sales_change'] = comp_data['sales_change_pct'].abs()
        main_row = comp_data.nlargest(1, 'abs_sales_change').iloc[0]
        
        sales_change = main_row['sales_change_pct']
        
        # Determine priority
        if abs(sales_change) > 25:
            insight_package['priority'] = 'critical'
        elif abs(sales_change) > 15:
            insight_package['priority'] = 'high'
        
        # Generate summary
        trend = "increased" if sales_change > 0 else "decreased"
        insight_package['summary'] = (
            f"Sales {trend} by {abs(sales_change):.1f}% compared to previous period"
        )
        
        # Add period comparison details - safely get region/segment
        comp_dict = main_row.to_dict()
        period_insight = self.generate_period_comparison_insight(
            comp_dict, 
            region=main_row.get('region') if 'region' in main_row else None,
            segment=main_row.get('segment') if 'segment' in main_row else None
        )
        insight_package['details'].append(period_insight)
        
        # Add root cause analysis
        if cause_analysis:
            cause_insight = self.generate_root_cause_insight(
                cause_analysis,
                region=main_row.get('region') if 'region' in main_row else None,
                segment=main_row.get('segment') if 'segment' in main_row else None
            )
            if cause_insight:
                insight_package['details'].append(cause_insight)
        
        # Add anomaly alerts
        for anomaly in anomalies[:3]:  # Limit to top 3
            anomaly_alert = self.generate_anomaly_alert(anomaly)
            insight_package['details'].append(anomaly_alert)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            sales_change, 
            cause_analysis,
            main_row.get('region') if 'region' in main_row else None
        )
        insight_package['recommendations'] = recommendations
        
        return insight_package
    
    def _generate_recommendations(self, sales_change: float,
                                 cause_analysis: Dict,
                                 region: str = None) -> List[str]:
        """
        Generate actionable recommendations based on insights.
        
        Args:
            sales_change (float): Sales change percentage
            cause_analysis (Dict): Root cause data
            region (str): Region name
        
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if sales_change < -15:
            recommendations.append(
                "💡 Consider launching targeted campaigns to boost sales"
            )
            
            # Check if leads are the issue
            for driver in cause_analysis.get('potential_drivers', []):
                if driver['driver'] == 'leads' and driver['change_pct'] < -10:
                    recommendations.append(
                        "💡 Increase lead generation activities and marketing spend"
                    )
                elif driver['driver'] == 'conversion_rate' and driver['change_pct'] < -10:
                    recommendations.append(
                        "💡 Review sales process and provide additional training to sales team"
                    )
            
            if region:
                recommendations.append(
                    f"💡 Conduct market analysis specific to {region} region"
                )
        
        elif sales_change > 15:
            recommendations.append(
                "💡 Analyze success factors and replicate in other regions"
            )
            recommendations.append(
                "💡 Ensure adequate inventory and resources to meet increased demand"
            )
        
        return recommendations
    
    def format_insight_report(self, insight_package: Dict) -> str:
        """
        Format insight package into readable report.
        
        Args:
            insight_package (Dict): Comprehensive insight data
        
        Returns:
            str: Formatted report
        """
        priority_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        
        emoji = priority_emoji.get(insight_package['priority'], '🟡')
        
        report = "\n" + "=" * 60 + "\n"
        report += f"{emoji} BUSINESS INSIGHT REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"📅 Generated: {insight_package['timestamp']}\n"
        report += f"🎯 Priority: {insight_package['priority'].upper()}\n\n"
        
        report += "📊 SUMMARY\n"
        report += "-" * 60 + "\n"
        report += insight_package['summary'] + "\n\n"
        
        if insight_package['details']:
            report += "📋 DETAILS\n"
            report += "-" * 60 + "\n"
            for detail in insight_package['details']:
                report += detail + "\n\n"
        
        if insight_package['recommendations']:
            report += "💡 RECOMMENDATIONS\n"
            report += "-" * 60 + "\n"
            for rec in insight_package['recommendations']:
                report += rec + "\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report


if __name__ == "__main__":
    # Test the insight engine
    engine = InsightEngine()
    
    # Sample comparison data
    comparison_data = {
        'sales_change_pct': -18.5,
        'sales_current': 950000,
        'sales_previous': 1170000
    }
    
    # Sample cause analysis
    cause_data = {
        'potential_drivers': [
            {'driver': 'leads', 'change_pct': -25.0, 'likely_cause': True},
            {'driver': 'conversion_rate', 'change_pct': -5.0, 'likely_cause': False}
        ]
    }
    
    # Sample anomaly
    anomaly = {
        'region': 'Chennai',
        'metric': 'sales',
        'deviation_pct': -22.5,
        'anomaly_type': 'below_average'
    }
    
    # Generate insights
    print(engine.generate_period_comparison_insight(comparison_data, region='Chennai', segment='Enterprise'))
    print("\n")
    print(engine.generate_root_cause_insight(cause_data, region='Chennai'))
    print("\n")
    print(engine.generate_anomaly_alert(anomaly))