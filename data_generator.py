"""
Data Generator for Mini Zia
Simulates realistic business KPI data with intentional anomalies for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_business_data(days=90, seed=42):
    """
    Generate simulated business data with realistic patterns and anomalies.
    
    Args:
        days (int): Number of days to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Business KPI data
    """
    np.random.seed(seed)
    
    # Configuration
    regions = ['Chennai', 'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad']
    segments = ['SMB', 'Enterprise']
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for date in dates:
        day_num = (date - start_date).days
        
        for region in regions:
            for segment in segments:
                
                # Base metrics with seasonal patterns
                base_sales = 100000 if segment == 'Enterprise' else 30000
                base_leads = 200 if segment == 'Enterprise' else 500
                
                # Add weekly seasonality (lower on weekends)
                weekday_factor = 0.7 if date.weekday() >= 5 else 1.0
                
                # Add growth trend
                growth_factor = 1 + (day_num * 0.002)
                
                # Regional multipliers
                region_multipliers = {
                    'Chennai': 1.0,
                    'Bangalore': 1.2,
                    'Mumbai': 1.3,
                    'Delhi': 1.1,
                    'Hyderabad': 0.9
                }
                region_factor = region_multipliers[region]
                
                # Calculate base metrics
                sales = base_sales * weekday_factor * growth_factor * region_factor
                leads = base_leads * weekday_factor * growth_factor * region_factor
                
                # Add random noise
                sales *= np.random.uniform(0.9, 1.1)
                leads *= np.random.uniform(0.85, 1.15)
                
                # Inject deliberate anomalies for testing
                
                # Anomaly 1: Chennai Enterprise leads dropped (days 75-85)
                if region == 'Chennai' and segment == 'Enterprise' and 75 <= day_num <= 85:
                    leads *= 0.6  # 40% drop
                    sales *= 0.75  # 25% drop due to fewer leads
                
                # Anomaly 2: Mumbai SMB spike (days 80-82)
                if region == 'Mumbai' and segment == 'SMB' and 80 <= day_num <= 82:
                    sales *= 1.5  # 50% spike
                    leads *= 1.3
                
                # Anomaly 3: Bangalore conversion rate drop (days 70-80)
                if region == 'Bangalore' and 70 <= day_num <= 80:
                    # Leads stay same but sales drop
                    sales *= 0.85
                
                # Calculate conversion rate
                conversion_rate = (sales / base_sales) / (leads / base_leads)
                conversion_rate = min(conversion_rate, 1.0)  # Cap at 100%
                
                # Round values
                sales = round(sales, 2)
                leads = int(leads)
                conversion_rate = round(conversion_rate * 100, 2)  # Convert to percentage
                
                data.append({
                    'date': date,
                    'region': region,
                    'segment': segment,
                    'sales': sales,
                    'leads': leads,
                    'conversion_rate': conversion_rate
                })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_recent_period(df, days=7):
    """
    Extract the most recent N days from the dataset.
    
    Args:
        df (pd.DataFrame): Full dataset
        days (int): Number of recent days
    
    Returns:
        pd.DataFrame: Recent period data
    """
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=days-1)
    return df[df['date'] >= cutoff_date].copy()


def get_previous_period(df, days=7, offset=7):
    """
    Extract the previous comparison period.
    
    Args:
        df (pd.DataFrame): Full dataset
        days (int): Number of days in period
        offset (int): Days before current period
    
    Returns:
        pd.DataFrame: Previous period data
    """
    max_date = df['date'].max()
    end_date = max_date - timedelta(days=offset)
    start_date = end_date - timedelta(days=days-1)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()


if __name__ == "__main__":
    # Test the data generator
    df = generate_business_data(days=90)
    
    print("=" * 60)
    print("📊 SAMPLE DATA GENERATED")
    print("=" * 60)
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nRegions: {df['region'].unique()}")
    print(f"Segments: {df['segment'].unique()}")
    
    print("\n" + "=" * 60)
    print("📋 SAMPLE ROWS")
    print("=" * 60)
    print(df.head(10))
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe())