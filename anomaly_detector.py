"""
Anomaly Detector for Mini Zia
Detects unusual patterns in business metrics using statistical methods.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class AnomalyDetector:
    """
    Detects anomalies in time-series business data using multiple techniques.
    """
    
    def __init__(self, threshold_std=2.0, threshold_pct=15.0):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold_std (float): Number of standard deviations for outlier detection
            threshold_pct (float): Percentage change threshold for significant changes
        """
        self.threshold_std = threshold_std
        self.threshold_pct = threshold_pct
    
    def detect_statistical_outliers(self, df: pd.DataFrame, metric: str, 
                                   group_by: List[str] = None) -> pd.DataFrame:
        """
        Detect outliers using rolling mean and standard deviation.
        
        Args:
            df (pd.DataFrame): Input data
            metric (str): Metric column to analyze
            group_by (List[str]): Columns to group by (e.g., ['region', 'segment'])
        
        Returns:
            pd.DataFrame: Data with anomaly flags
        """
        df = df.copy()
        df = df.sort_values('date')
        
        if group_by:
            # Calculate rolling statistics per group
            df['rolling_mean'] = df.groupby(group_by)[metric].transform(
                lambda x: x.rolling(window=7, min_periods=3).mean()
            )
            df['rolling_std'] = df.groupby(group_by)[metric].transform(
                lambda x: x.rolling(window=7, min_periods=3).std()
            )
        else:
            # Calculate overall rolling statistics
            df['rolling_mean'] = df[metric].rolling(window=7, min_periods=3).mean()
            df['rolling_std'] = df[metric].rolling(window=7, min_periods=3).std()
        
        # Detect outliers
        df['z_score'] = (df[metric] - df['rolling_mean']) / (df['rolling_std'] + 1e-6)
        df['is_outlier'] = np.abs(df['z_score']) > self.threshold_std
        
        return df
    
    def detect_significant_changes(self, current_df: pd.DataFrame,
                                   previous_df: pd.DataFrame,
                                   metrics: List[str],
                                   group_by: List[str] = None,
                                   mean_metrics: List[str] = None) -> pd.DataFrame:
        """
        Detect significant period-over-period changes.

        Args:
            current_df (pd.DataFrame): Current period data
            previous_df (pd.DataFrame): Previous period data
            metrics (List[str]): Metrics to compare
            group_by (List[str]): Grouping columns
            mean_metrics (List[str]): Metrics aggregated by mean (rates/%) instead of sum

        Returns:
            pd.DataFrame: Comparison results with anomaly flags
        """
        group_cols = group_by if group_by else []
        mean_set = set(mean_metrics or [])
        agg_map = {m: ('mean' if m in mean_set else 'sum') for m in metrics}

        if group_cols:
            current_agg = current_df.groupby(group_cols).agg(agg_map).reset_index()
            previous_agg = previous_df.groupby(group_cols).agg(agg_map).reset_index()
            comparison = current_agg.merge(previous_agg, on=group_cols, suffixes=('_current', '_prev'))
        else:
            comparison = pd.DataFrame()
            for metric in metrics:
                cur = current_df[metric].mean() if metric in mean_set else current_df[metric].sum()
                prev = previous_df[metric].mean() if metric in mean_set else previous_df[metric].sum()
                comparison[f'{metric}_current'] = [cur]
                comparison[f'{metric}_prev'] = [prev]

        # Calculate percentage changes
        for metric in metrics:
            comparison[f'{metric}_change_pct'] = (
                (comparison[f'{metric}_current'] - comparison[f'{metric}_prev']) /
                (comparison[f'{metric}_prev'] + 1e-6) * 100
            )
            comparison[f'{metric}_is_anomaly'] = (
                np.abs(comparison[f'{metric}_change_pct']) > self.threshold_pct
            )

        return comparison

    def find_regional_anomalies(self, df: pd.DataFrame, metric: str,
                               period_days: int = 7,
                               dimension: str = 'region') -> List[Dict]:
        """
        Identify groups (within a dimension) whose metric value is unusual vs the average.

        Args:
            df (pd.DataFrame): Input data
            metric (str): Metric to analyze
            period_days (int): Number of recent days to analyze
            dimension (str): Column to group across (e.g. 'region', 'product', 'channel')

        Returns:
            List[Dict]: List of anomalies
        """
        recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=period_days)]
        if dimension not in recent_data.columns or len(recent_data) == 0:
            return []

        agg = recent_data.groupby(dimension)[metric].sum().reset_index()
        n_groups = recent_data[dimension].nunique()
        overall_avg = (recent_data[metric].sum() / n_groups) if n_groups else 0

        anomalies = []
        if overall_avg == 0:
            return anomalies

        for _, row in agg.iterrows():
            pct_diff = ((row[metric] - overall_avg) / overall_avg) * 100
            if abs(pct_diff) > self.threshold_pct:
                anomalies.append({
                    'group': row[dimension],
                    'dimension': dimension,
                    'region': row[dimension],  # backward-compatible alias
                    'metric': metric,
                    'value': row[metric],
                    'avg_value': overall_avg,
                    'deviation_pct': round(pct_diff, 2),
                    'anomaly_type': 'above_average' if pct_diff > 0 else 'below_average'
                })

        return anomalies
    
    def analyze_segment_split(self, current_df: pd.DataFrame, 
                              previous_df: pd.DataFrame,
                              metric: str = 'sales',
                              region: str = None) -> Dict:
        """
        Analyze changes in segment contribution (SMB vs Enterprise).
        
        Args:
            current_df (pd.DataFrame): Current period data
            previous_df (pd.DataFrame): Previous period data
            metric (str): Metric to analyze
            region (str): Optional region filter
        
        Returns:
            Dict: Segment analysis results
        """
        if region:
            current_df = current_df[current_df['region'] == region]
            previous_df = previous_df[previous_df['region'] == region]
        
        current_seg = current_df.groupby('segment')[metric].sum()
        previous_seg = previous_df.groupby('segment')[metric].sum()
        
        results = {}
        
        for segment in ['SMB', 'Enterprise']:
            if segment in current_seg.index and segment in previous_seg.index:
                change_pct = ((current_seg[segment] - previous_seg[segment]) / 
                             previous_seg[segment]) * 100
                
                results[segment] = {
                    'current_value': current_seg[segment],
                    'previous_value': previous_seg[segment],
                    'change_pct': round(change_pct, 2),
                    'is_significant': abs(change_pct) > self.threshold_pct
                }
        
        return results


def format_anomaly_report(anomalies: List[Dict]) -> str:
    """
    Format detected anomalies into a readable report.
    
    Args:
        anomalies (List[Dict]): List of anomaly dictionaries
    
    Returns:
        str: Formatted report
    """
    if not anomalies:
        return "✅ No significant anomalies detected."
    
    report = "🚨 DETECTED ANOMALIES\n"
    report += "=" * 60 + "\n\n"
    
    for i, anomaly in enumerate(anomalies, 1):
        report += f"{i}. {anomaly['region']} - {anomaly['metric'].upper()}\n"
        report += f"   Value: {anomaly['value']:,.2f}\n"
        report += f"   Average: {anomaly['avg_value']:,.2f}\n"
        report += f"   Deviation: {anomaly['deviation_pct']:+.1f}%\n"
        report += f"   Type: {anomaly['anomaly_type'].replace('_', ' ').title()}\n\n"
    
    return report


if __name__ == "__main__":
    from data_generator import generate_business_data, get_recent_period, get_previous_period
    
    # Generate test data
    df = generate_business_data(days=90)
    current = get_recent_period(df, days=7)
    previous = get_previous_period(df, days=7, offset=7)
    
    # Initialize detector
    detector = AnomalyDetector(threshold_std=2.0, threshold_pct=15.0)
    
    # Test 1: Period-over-period changes
    print("=" * 60)
    print("📊 PERIOD-OVER-PERIOD ANALYSIS")
    print("=" * 60)
    comparison = detector.detect_significant_changes(
        current, previous, 
        metrics=['sales', 'leads'], 
        group_by=['region', 'segment']
    )
    print(comparison[comparison['sales_is_anomaly'] | comparison['leads_is_anomaly']])
    
    # Test 2: Regional anomalies
    print("\n" + "=" * 60)
    print("🗺️ REGIONAL ANOMALIES")
    print("=" * 60)
    regional_anomalies = detector.find_regional_anomalies(current, metric='sales')
    print(format_anomaly_report(regional_anomalies))