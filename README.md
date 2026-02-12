# 🤖 Mini Zia: Automated Business Insight Generator

> **Interview-Ready Project for Zoho Data Scientist Position**

A production-grade system that automatically converts raw KPI data into actionable business insights using deterministic analytics and optional LLM enhancement.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Technical Details](#technical-details)
7. [Interview Discussion Points](#interview-discussion-points)

---

## 🎯 Overview

### Problem Statement

Business users struggle to interpret dashboards and KPIs. They need automatic explanations like:

> *"Sales dropped 18% in Chennai due to reduced enterprise leads."*

### Solution

Mini Zia detects anomalies, analyzes trends, identifies root causes, and explains them in simple English using:

- **Deterministic Analytics**: Rule-based, explainable AI
- **Statistical Methods**: Rolling means, standard deviations, percentage changes
- **Natural Language Generation**: Template-based insights
- **Optional LLM Layer**: Conversational enhancement (Claude/GPT)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Mini Zia                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ Data         │──────▶│ Anomaly      │                   │
│  │ Generator    │      │ Detector     │                   │
│  └──────────────┘      └──────┬───────┘                   │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                   │
│                        │ Trend        │                   │
│                        │ Analyzer     │                   │
│                        └──────┬───────┘                   │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                   │
│                        │ Insight      │                   │
│                        │ Engine       │                   │
│                        └──────┬───────┘                   │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                   │
│                        │ LLM          │ (Optional)        │
│                        │ Rephraser    │                   │
│                        └──────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Module | Purpose | Key Algorithms |
|--------|---------|----------------|
| **data_generator.py** | Simulate realistic business data | Time-series with seasonality, deliberate anomalies |
| **anomaly_detector.py** | Detect unusual patterns | Rolling statistics (μ ± 2σ), percentage thresholds |
| **trend_analyzer.py** | Compare periods & find drivers | Period-over-period analysis, segment decomposition |
| **insight_engine.py** | Generate natural language | Template-based NLG with business rules |
| **llm_rephraser.py** | Enhance with conversational AI | Optional API integration (Claude/GPT) |
| **main.py** | Orchestrate the pipeline | End-to-end workflow management |

---

## ✨ Features

### 1. Anomaly Detection

- **Statistical Outlier Detection**: Uses rolling mean ± 2σ
- **Threshold-Based Alerts**: Flags changes > 15%
- **Regional Comparisons**: Identifies underperforming regions
- **Segment Analysis**: SMB vs Enterprise breakdown

### 2. Trend Analysis

- **Period-over-Period**: Compare current vs previous week/month
- **Root Cause Identification**: Correlate metrics (leads → sales)
- **Top Contributors**: Rank regions/segments by impact
- **Trend Direction**: Detect increasing/decreasing/stable patterns

### 3. Natural Language Generation

- **Template-Based Insights**: Structured, explainable outputs
- **Priority Classification**: Critical / High / Medium / Low
- **Actionable Recommendations**: Context-aware suggestions
- **Executive Summaries**: High-level overviews

### 4. LLM Enhancement (Optional)

- **Tone Adaptation**: Professional / Casual / Executive
- **Conversational Rephrasing**: Make insights more human
- **Batch Processing**: Enhance multiple insights efficiently
- **API Abstraction**: Support for Claude & GPT

---

## 🚀 Installation & Setup

### Prerequisites

```bash
Python 3.8+
pip install pandas numpy
```

### Optional (for LLM enhancement)

```bash
pip install anthropic  # For Claude
pip install openai     # For GPT
```

### Project Structure

```
mini_zia/
├── data_generator.py
├── anomaly_detector.py
├── trend_analyzer.py
├── insight_engine.py
├── llm_rephraser.py
├── main.py
└── README.md
```

### Quick Start

```python
# Clone or create the project directory
cd mini_zia

# Run the main demo
python main.py
```

---

## 📖 Usage Guide

### Basic Usage

```python
from main import MiniZia
from data_generator import generate_business_data

# Generate data
df = generate_business_data(days=90)

# Initialize Mini Zia
zia = MiniZia(
    anomaly_threshold_std=2.0,      # 2 standard deviations
    anomaly_threshold_pct=15.0,     # 15% change threshold
    significance_threshold=10.0,     # 10% for significance
    enable_llm=False                 # Toggle LLM enhancement
)

# Run analysis
results = zia.analyze(
    df=df,
    current_period_days=7,
    previous_period_days=7,
    region_filter=None,              # Optional: 'Chennai'
    segment_filter=None              # Optional: 'Enterprise'
)
```

### Output Example

```
🔴 BUSINESS INSIGHT REPORT
============================================================

📅 Generated: 2026-01-06 14:30:22
🎯 Priority: HIGH

📊 SUMMARY
------------------------------------------------------------
Sales decreased by 18.5% compared to previous period

📋 DETAILS
------------------------------------------------------------
⚠️ Sales dropped by 18.5% in Chennai (Enterprise)
   Current: ₹950,000 ↘️
   Priority: HIGH

🔍 Primary cause in Chennai (Enterprise): Leads declined by 25.0%

📉 Anomaly Alert: Chennai
   Sales is 22.5% below average
   Requires investigation

💡 RECOMMENDATIONS
------------------------------------------------------------
💡 Consider launching targeted campaigns to boost sales
💡 Increase lead generation activities and marketing spend
💡 Conduct market analysis specific to Chennai region
```

### Advanced: Region-Specific Analysis

```python
# Deep dive into Chennai
results_chennai = zia.analyze(
    df=df,
    region_filter='Chennai',
    current_period_days=7
)
```

### Advanced: LLM Enhancement

```python
from llm_rephraser import LLMRephraser

rephraser = LLMRephraser(api_key="your_api_key", provider="anthropic")

enhanced = rephraser.enhance_insight(
    insight_text=results['insights']['summary'],
    tone='executive'
)
```

---

## 🔬 Technical Details

### Anomaly Detection Algorithm

```python
# Rolling statistics approach
rolling_mean = data.rolling(window=7).mean()
rolling_std = data.rolling(window=7).std()
z_score = (value - rolling_mean) / rolling_std

# Flag if |z_score| > 2 (outside 95% confidence interval)
is_anomaly = abs(z_score) > 2.0
```

### Root Cause Analysis

```python
# Identify correlations
sales_change = (current_sales - prev_sales) / prev_sales
leads_change = (current_leads - prev_leads) / prev_leads

# Same direction + significant = likely cause
is_likely_cause = (
    (sales_change * leads_change > 0) and  # Same direction
    (abs(leads_change) > 0.10)             # Significant change
)
```

### Insight Generation Templates

```python
templates = {
    'decline': "{metric} dropped by {pct}% in {region}",
    'spike': "{metric} increased by {pct}% in {region}",
    'cause': "Primary cause: {driver} changed by {pct}%"
}
```

---

## 🎤 Interview Discussion Points

### 1. Design Choices

**Q: Why rule-based over pure ML?**

**A:** 
- **Explainability**: Business users need to understand "why"
- **Reliability**: No training data requirements, no drift
- **Speed**: Real-time analysis without model inference
- **Zoho Philosophy**: Zia uses deterministic logic for transparency

**Q: When would you add ML?**

**A:**
- Forecasting future trends (ARIMA, Prophet)
- Anomaly detection in high-dimensional data (Isolation Forest)
- Personalized insights based on user behavior
- But always keep rule-based as fallback for explainability

### 2. Scalability Considerations

**Current Design**: In-memory pandas (good for demo)

**Production Enhancements**:
- **Data Layer**: Replace with SQL queries (PostgreSQL, ClickHouse)
- **Caching**: Redis for frequently accessed aggregations
- **Batch Processing**: Apache Spark for large-scale analytics
- **Real-time**: Apache Kafka + Stream processing

### 3. LLM Integration Strategy

**Why LLM is Enhancement, Not Core**:
- Deterministic logic ensures consistency
- LLM adds conversational polish
- Falls back gracefully if API fails
- Cost-effective (only rephrase, not analyze)

**Production LLM Best Practices**:
- Cache rephrased insights by hash
- Use lower temperature (0.3) for consistency
- Implement rate limiting
- Monitor token usage and costs

### 4. Business Value

**Measurable Impact**:
- Reduce time to insight from hours to seconds
- Increase dashboard engagement by surfacing key findings
- Enable non-technical users to be data-driven
- Proactive alerting prevents revenue loss

### 5. Testing Strategy

**Unit Tests**:
```python
def test_anomaly_detection():
    df = create_test_data_with_spike()
    detector = AnomalyDetector(threshold_std=2.0)
    anomalies = detector.detect_outliers(df)
    assert len(anomalies) > 0
```

**Integration Tests**:
- Test full pipeline with known anomaly
- Verify insight accuracy
- Check performance with large datasets

**A/B Testing**:
- Compare user engagement with/without insights
- Measure time to decision-making
- Track alert accuracy (true positives vs false positives)

### 6. Future Enhancements

1. **Multi-metric Analysis**: Analyze correlations across 10+ KPIs
2. **Predictive Insights**: "Sales likely to drop 15% next week"
3. **Automated Actions**: Trigger campaigns based on insights
4. **Personalization**: Different insights for different roles
5. **Interactive Exploration**: Let users drill down into causes

---

## 📊 Sample Output Walkthrough

### Scenario: Chennai Enterprise Sales Drop

**Input Data** (Last 7 days vs Previous 7 days):

| Region | Segment | Current Sales | Previous Sales | Change |
|--------|---------|--------------|----------------|--------|
| Chennai | Enterprise | ₹950,000 | ₹1,170,000 | -18.8% |

**Analysis Pipeline**:

1. **Anomaly Detection**: Flags Chennai as 22% below average
2. **Trend Analysis**: Identifies -25% drop in leads
3. **Root Cause**: Correlates lead decline with sales drop
4. **Insight Generation**: Creates structured natural language output
5. **Recommendations**: Suggests specific actions

**Final Insight**:

> ⚠️ **Sales Alert – Chennai (Enterprise)**
>
> Sales dropped by 18.5% compared to last week.
>
> **Primary cause**: Enterprise leads declined by 25%.
>
> **Recommended actions**:
> - Increase lead generation activities and marketing spend
> - Conduct market analysis specific to Chennai region
> - Review competitive landscape for Enterprise segment

---

## 🏆 Why This Project Stands Out

1. **Production-Ready Code**
   - Modular, documented, PEP8 compliant
   - Error handling and logging
   - Extensible architecture

2. **Business-Centric**
   - Solves real problems non-technical users face
   - Generates actionable insights, not just numbers
   - Aligns with Zoho's product philosophy

3. **Technical Depth**
   - Statistical rigor (z-scores, rolling windows)
   - Smart design choices (rules + optional LLM)
   - Scalability considerations

4. **Interview-Ready**
   - Clear talking points for each component
   - Demonstrates both coding and product thinking
   - Shows understanding of Zoho's AI assistant space

---

## 📞 Contact & Feedback

**Created By**: Sinwan_siraj 

**Key Technologies**: Python, Pandas, NumPy, Statistical Analysis, NLG, LLM Integration

**Demonstrates**: Data Analytics, Business Intelligence, Explainable AI, Product Thinking

---

## 📄 License

This is a portfolio project created for interview purposes.

---

**Built with ❤️ for Zoho**
