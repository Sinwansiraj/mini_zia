"""
Trend Analyzer for Mini Zia
Analyzes trends and identifies root causes of metric changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TrendAnalyzer:
    """
    Analyzes business trends and identifies contributing factors.
    """
    
    def __init__(self, significance_threshold=10.0):
        """
        Initialize the trend analyzer.
        
        Args:
            significance_threshold (float): Minimum percentage change to consider significant
        """
        self.significance_threshold = significance_threshold
    
    def compare_periods(self, current_df: pd.DataFrame, 
                       previous_df: pd.DataFrame,
                       group_by: List[str] = None) -> pd.DataFrame:
        """
        Compare two time periods with detailed breakdowns.
        
        Args:
            current_df (pd.DataFrame): Current period data
            previous_df (pd.DataFrame): Previous period data
            group_by (List[str]): Columns to group by
        
        Returns:
            pd.DataFrame: Detailed comparison
        """
        metrics = ['sales', 'leads', 'conversion_rate']
        group_cols = group_by if group_by else []
        
        # Aggregate data
        if group_cols:
            current_agg = current_df.groupby(group_cols).agg({
                'sales': 'sum',
                'leads': 'sum',
                'conversion_rate': 'mean'
            }).reset_index()
            
            previous_agg = previous_df.groupby(group_cols).agg({
                'sales': 'sum',
                'leads': 'sum',
                'conversion_rate': 'mean'
            }).reset_index()
            
            comparison = current_agg.merge(previous_agg, on=group_cols, 
                                          suffixes=('_current', '_previous'))
        else:
            current_agg = pd.DataFrame({
                'sales': [current_df['sales'].sum()],
                'leads': [current_df['leads'].sum()],
                'conversion_rate': [current_df['conversion_rate'].mean()]
            })
            
            previous_agg = pd.DataFrame({
                'sales': [previous_df['sales'].sum()],
                'leads': [previous_df['leads'].sum()],
                'conversion_rate': [previous_df['conversion_rate'].mean()]
            })
            
            comparison = pd.DataFrame()
            for metric in metrics:
                comparison[f'{metric}_current'] = current_agg[metric]
                comparison[f'{metric}_previous'] = previous_agg[metric]
        
        # Calculate changes
        for metric in metrics:
            comparison[f'{metric}_change'] = (
                comparison[f'{metric}_current'] - comparison[f'{metric}_previous']
            )
            comparison[f'{metric}_change_pct'] = (
                comparison[f'{metric}_change'] / 
                (comparison[f'{metric}_previous'] + 1e-6) * 100
            )
        
        return comparison
    
    def identify_top_contributors(self, comparison_df: pd.DataFrame,
                                 metric: str = 'sales',
                                 top_n: int = 3) -> List[Dict]:
        """
        Identify the top contributors to metric changes.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe from compare_periods
            metric (str): Metric to analyze
            top_n (int): Number of top contributors to return
        
        Returns:
            List[Dict]: Top contributors with details
        """
        change_col = f'{metric}_change'
        change_pct_col = f'{metric}_change_pct'
        
        if change_col not in comparison_df.columns:
            return []
        
        # Sort by absolute change
        sorted_df = comparison_df.copy()
        sorted_df['abs_change'] = sorted_df[change_col].abs()
        sorted_df = sorted_df.sort_values('abs_change', ascending=False)
        
        contributors = []
        
        for _, row in sorted_df.head(top_n).iterrows():
            contributor = {
                'change': row[change_col],
                'change_pct': row[change_pct_col],
                'current_value': row[f'{metric}_current'],
                'previous_value': row[f'{metric}_previous']
            }
            
            # Add grouping information
            if 'region' in row:
                contributor['region'] = row['region']
            if 'segment' in row:
                contributor['segment'] = row['segment']
            
            contributors.append(contributor)
        
        return contributors
    
    def analyze_cause_effect(self, current_df: pd.DataFrame,
                            previous_df: pd.DataFrame,
                            target_metric: str = 'sales',
                            driver_metrics: List[str] = None) -> Dict:
        """
        Analyze relationships between metrics to identify root causes.
        
        Args:
            current_df (pd.DataFrame): Current period data
            previous_df (pd.DataFrame): Previous period data
            target_metric (str): The metric that changed
            driver_metrics (List[str]): Potential driver metrics
        
        Returns:
            Dict: Analysis of potential causes
        """
        if driver_metrics is None:
            driver_metrics = ['leads', 'conversion_rate']
        
        # Calculate overall changes
        target_current = current_df[target_metric].sum()
        target_previous = previous_df[target_metric].sum()
        target_change_pct = ((target_current - target_previous) / target_previous) * 100
        
        causes = {
            'target_metric': target_metric,
            'target_change_pct': round(target_change_pct, 2),
            'potential_drivers': []
        }
        
        for driver in driver_metrics:
            if driver == 'conversion_rate':
                driver_current = current_df[driver].mean()
                driver_previous = previous_df[driver].mean()
            else:
                driver_current = current_df[driver].sum()
                driver_previous = previous_df[driver].sum()
            
            driver_change_pct = ((driver_current - driver_previous) / driver_previous) * 100
            
            # Determine if this is a significant driver
            is_significant = abs(driver_change_pct) > self.significance_threshold
            
            # Check correlation direction
            same_direction = (target_change_pct * driver_change_pct) > 0
            
            causes['potential_drivers'].append({
                'driver': driver,
                'change_pct': round(driver_change_pct, 2),
                'is_significant': is_significant,
                'correlation': 'positive' if same_direction else 'negative',
                'likely_cause': is_significant and same_direction
            })
        
        return causes
    
    def get_segment_breakdown(self, df: pd.DataFrame, 
                             metric: str = 'sales',
                             region: str = None) -> Dict:
        """
        Break down metrics by segment (SMB vs Enterprise).
        
        Args:
            df (pd.DataFrame): Input data
            metric (str): Metric to analyze
            region (str): Optional region filter
        
        Returns:
            Dict: Segment breakdown
        """
        if region:
            df = df[df['region'] == region]
        
        segment_data = df.groupby('segment')[metric].sum().to_dict()
        
        total = sum(segment_data.values())
        
        breakdown = {
            'total': total,
            'segments': {}
        }
        
        for segment, value in segment_data.items():
            contribution_pct = (value / total * 100) if total > 0 else 0
            breakdown['segments'][segment] = {
                'value': value,
                'contribution_pct': round(contribution_pct, 2)
            }
        
        return breakdown
    
    def detect_trends(self, df: pd.DataFrame, metric: str = 'sales',
                     window: int = 7) -> Dict:
        """
        Detect trend direction over time using linear regression approach.
        
        Args:
            df (pd.DataFrame): Time-series data
            metric (str): Metric to analyze
            window (int): Window for trend calculation
        
        Returns:
            Dict: Trend information
        """
        df = df.sort_values('date')
        recent_data = df.tail(window)
        
        if len(recent_data) < 3:
            return {'trend': 'insufficient_data'}
        
        # Simple trend calculation using first and last values
        values = recent_data.groupby('date')[metric].sum().values
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        first_half_avg = np.mean(values[:len(values)//2])
        second_half_avg = np.mean(values[len(values)//2:])
        
        pct_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if abs(pct_change) < 5:
            trend = 'stable'
        elif pct_change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'change_pct': round(pct_change, 2),
            'first_half_avg': round(first_half_avg, 2),
            'second_half_avg': round(second_half_avg, 2)
        }


if __name__ == "__main__":
    from data_generator import generate_business_data, get_recent_period, get_previous_period
    
    # Generate test data
    df = generate_business_data(days=90)
    current = get_recent_period(df, days=7)
    previous = get_previous_period(df, days=7, offset=7)
    
    # Initialize analyzer
    analyzer = TrendAnalyzer(significance_threshold=10.0)
    
    # Test 1: Compare periods
    print("=" * 60)
    print("📊 PERIOD COMPARISON")
    print("=" * 60)
    comparison = analyzer.compare_periods(current, previous, group_by=['region'])
    print(comparison[['region', 'sales_change_pct', 'leads_change_pct']])
    
    # Test 2: Identify top contributors
    print("\n" + "=" * 60)
    print("🎯 TOP CONTRIBUTORS TO SALES CHANGE")
    print("=" * 60)
    contributors = analyzer.identify_top_contributors(comparison, metric='sales', top_n=3)
    for i, contrib in enumerate(contributors, 1):
        print(f"{i}. {contrib['region']}: {contrib['change_pct']:+.1f}% change")
    
    # Test 3: Cause-effect analysis
    print("\n" + "=" * 60)
    print("🔍 ROOT CAUSE ANALYSIS")
    print("=" * 60)
    causes = analyzer.analyze_cause_effect(current, previous, target_metric='sales')
    print(f"Sales changed by: {causes['target_change_pct']:+.1f}%")
    for driver in causes['potential_drivers']:
        if driver['likely_cause']:
            print(f"  → Likely cause: {driver['driver']} changed by {driver['change_pct']:+.1f}%")