"""
Mini Zia - Automated Business Insight Generator
Main orchestrator that brings all components together.
"""

import pandas as pd
from datetime import datetime
import sys

# Import all modules
from data_generator import (
    generate_business_data, 
    get_recent_period, 
    get_previous_period
)
from anomaly_detector import AnomalyDetector, format_anomaly_report
from trend_analyzer import TrendAnalyzer
from insight_engine import InsightEngine
from llm_rephraser import LLMRephraser


class MiniZia:
    """
    Main orchestrator for the automated business insight generator.
    """
    
    def __init__(self, 
                 anomaly_threshold_std=2.0,
                 anomaly_threshold_pct=15.0,
                 significance_threshold=10.0,
                 enable_llm=False,
                 llm_api_key=None):
        """
        Initialize Mini Zia with all components.
        
        Args:
            anomaly_threshold_std (float): Standard deviation threshold for anomalies
            anomaly_threshold_pct (float): Percentage threshold for anomalies
            significance_threshold (float): Significance threshold for trends
            enable_llm (bool): Enable LLM enhancement
            llm_api_key (str): API key for LLM provider
        """
        self.anomaly_detector = AnomalyDetector(
            threshold_std=anomaly_threshold_std,
            threshold_pct=anomaly_threshold_pct
        )
        self.trend_analyzer = TrendAnalyzer(
            significance_threshold=significance_threshold
        )
        self.insight_engine = InsightEngine()
        self.llm_rephraser = LLMRephraser(
            api_key=llm_api_key if enable_llm else None
        )
        self.enable_llm = enable_llm
    
    def analyze(self, 
                df: pd.DataFrame,
                current_period_days: int = 7,
                previous_period_days: int = 7,
                region_filter: str = None,
                segment_filter: str = None) -> dict:
        """
        Run complete analysis pipeline.
        
        Args:
            df (pd.DataFrame): Input business data
            current_period_days (int): Days in current period
            previous_period_days (int): Days in previous period
            region_filter (str): Optional region filter
            segment_filter (str): Optional segment filter
        
        Returns:
            dict: Complete analysis results
        """
        print("\n🚀 Starting Mini Zia Analysis...")
        print("=" * 60)
        
        # Extract time periods
        current = get_recent_period(df, days=current_period_days)
        previous = get_previous_period(df, days=previous_period_days, 
                                      offset=current_period_days)
        
        print(f"📅 Current Period: {current['date'].min()} to {current['date'].max()}")
        print(f"📅 Previous Period: {previous['date'].min()} to {previous['date'].max()}")
        print(f"📊 Records: Current={len(current)}, Previous={len(previous)}")
        
        # Apply filters if specified
        if region_filter:
            current = current[current['region'] == region_filter]
            previous = previous[previous['region'] == region_filter]
            print(f"🗺️  Filtered by region: {region_filter}")
        
        if segment_filter:
            current = current[current['segment'] == segment_filter]
            previous = previous[previous['segment'] == segment_filter]
            print(f"🎯 Filtered by segment: {segment_filter}")
        
        results = {
            'timestamp': datetime.now(),
            'filters': {
                'region': region_filter,
                'segment': segment_filter
            },
            'anomalies': [],
            'comparisons': None,
            'root_causes': None,
            'insights': None
        }
        
        # Step 1: Detect anomalies
        print("\n🔍 Step 1: Detecting Anomalies...")
        print("-" * 60)
        
        regional_anomalies = self.anomaly_detector.find_regional_anomalies(
            current, 
            metric='sales',
            period_days=current_period_days
        )
        results['anomalies'] = regional_anomalies
        
        if regional_anomalies:
            print(format_anomaly_report(regional_anomalies))
        else:
            print("✅ No significant anomalies detected")
        
        # Step 2: Period comparison
        print("\n📊 Step 2: Analyzing Period-over-Period Changes...")
        print("-" * 60)
        
        group_by = []
        if not region_filter:
            group_by.append('region')
        if not segment_filter:
            group_by.append('segment')
        
        if not group_by:
            group_by = None
        
        comparison = self.anomaly_detector.detect_significant_changes(
            current, 
            previous,
            metrics=['sales', 'leads', 'conversion_rate'],
            group_by=group_by
        )
        results['comparisons'] = comparison
        
        # Display significant changes
        significant = comparison[
            comparison['sales_is_anomaly'] | 
            comparison['leads_is_anomaly']
        ]
        
        if len(significant) > 0:
            print("⚠️  Significant Changes Detected:")
            display_cols = [col for col in ['region', 'segment', 'sales_change_pct', 
                           'leads_change_pct', 'conversion_rate_change_pct'] 
                           if col in significant.columns]
            print(significant[display_cols].to_string(index=False))
        else:
            print("✅ No significant period-over-period changes")
        
        # Step 3: Root cause analysis
        print("\n🔍 Step 3: Identifying Root Causes...")
        print("-" * 60)
        
        # Analyze for most significant region/segment combo
        if len(comparison) > 0:
            comparison['abs_sales_change'] = comparison['sales_change_pct'].abs()
            top_change = comparison.nlargest(1, 'abs_sales_change').iloc[0]
            
            # Filter data for root cause analysis
            rca_current = current
            rca_previous = previous
            
            if 'region' in top_change:
                rca_region = top_change['region']
                rca_current = rca_current[rca_current['region'] == rca_region]
                rca_previous = rca_previous[rca_previous['region'] == rca_region]
            
            if 'segment' in top_change:
                rca_segment = top_change['segment']
                rca_current = rca_current[rca_current['segment'] == rca_segment]
                rca_previous = rca_previous[rca_previous['segment'] == rca_segment]
            
            root_causes = self.trend_analyzer.analyze_cause_effect(
                rca_current,
                rca_previous,
                target_metric='sales',
                driver_metrics=['leads', 'conversion_rate']
            )
            results['root_causes'] = root_causes
            
            print(f"Sales Change: {root_causes['target_change_pct']:+.1f}%")
            print("\nPotential Drivers:")
            for driver in root_causes['potential_drivers']:
                status = "✓ LIKELY CAUSE" if driver['likely_cause'] else "✗ Not significant"
                print(f"  • {driver['driver']}: {driver['change_pct']:+.1f}% [{status}]")
        
        # Step 4: Generate insights
        print("\n💡 Step 4: Generating Natural Language Insights...")
        print("-" * 60)
        
        insight_package = self.insight_engine.generate_comprehensive_insight(
            comparison=comparison,
            cause_analysis=results.get('root_causes'),
            anomalies=regional_anomalies,
            region=region_filter,
            segment=segment_filter
        )
        results['insights'] = insight_package
        
        # Format and display
        report = self.insight_engine.format_insight_report(insight_package)
        print(report)
        
        # Step 5: LLM Enhancement (optional)
        if self.enable_llm:
            print("\n🤖 Step 5: Enhancing with LLM...")
            print("-" * 60)
            
            enhanced_summary = self.llm_rephraser.enhance_insight(
                insight_package['summary'],
                tone='professional'
            )
            print(f"Enhanced Summary: {enhanced_summary}")
        
        return results
    
    def generate_summary_metrics(self, results: dict) -> dict:
        """
        Generate summary metrics from analysis results.
        
        Args:
            results (dict): Analysis results
        
        Returns:
            dict: Summary metrics
        """
        summary = {
            'total_anomalies': len(results['anomalies']),
            'critical_issues': 0,
            'priority': results['insights']['priority'] if results['insights'] else 'medium'
        }
        
        if results['comparisons'] is not None:
            comparison = results['comparisons']
            summary['regions_analyzed'] = len(comparison) if 'region' in comparison.columns else 1
            summary['significant_changes'] = len(
                comparison[comparison['sales_is_anomaly'] | comparison['leads_is_anomaly']]
            )
        
        return summary


def main():
    """
    Main execution function - demonstrates complete workflow.
    """
    print("\n" + "=" * 60)
    print("🤖 MINI ZIA - Automated Business Insight Generator")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n📊 Generating Business Data...")
    df = generate_business_data(days=90)
    print(f"✓ Generated {len(df)} records across {df['date'].nunique()} days")
    
    # Step 2: Initialize Mini Zia
    print("\n🔧 Initializing Mini Zia...")
    zia = MiniZia(
        anomaly_threshold_std=2.0,
        anomaly_threshold_pct=15.0,
        significance_threshold=10.0,
        enable_llm=False  # Set to True and provide API key for LLM enhancement
    )
    print("✓ All components initialized")
    
    # Step 3: Run full analysis
    print("\n" + "=" * 60)
    print("🎯 SCENARIO 1: Overall Business Analysis")
    print("=" * 60)
    
    results_overall = zia.analyze(df, current_period_days=7, previous_period_days=7)
    
    # Step 4: Analyze specific region (Chennai - where we injected anomaly)
    print("\n\n" + "=" * 60)
    print("🎯 SCENARIO 2: Chennai Region Deep Dive")
    print("=" * 60)
    
    results_chennai = zia.analyze(
        df, 
        current_period_days=7, 
        previous_period_days=7,
        region_filter='Chennai'
    )
    
    # Step 5: Analyze specific segment
    print("\n\n" + "=" * 60)
    print("🎯 SCENARIO 3: Enterprise Segment Analysis")
    print("=" * 60)
    
    results_enterprise = zia.analyze(
        df,
        current_period_days=7,
        previous_period_days=7,
        segment_filter='Enterprise'
    )
    
    # Step 6: Generate executive summary
    print("\n\n" + "=" * 60)
    print("📈 EXECUTIVE SUMMARY")
    print("=" * 60)
    
    summary = zia.generate_summary_metrics(results_overall)
    print(f"\n✓ Analyzed {summary.get('regions_analyzed', 0)} regions")
    print(f"✓ Detected {summary['total_anomalies']} anomalies")
    print(f"✓ Found {summary.get('significant_changes', 0)} significant changes")
    print(f"✓ Overall Priority: {summary['priority'].upper()}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results_overall, results_chennai, results_enterprise


if __name__ == "__main__":
    try:
        results = main()
        print("\n💡 TIP: Customize thresholds, filters, and enable LLM enhancement")
        print("    for production use. See llm_rephraser.py for API integration guide.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)