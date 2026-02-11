
# Earnings Call Analyzer

CLI-based tool that analyzes earnings call transcripts and generates structured financial metrics, sentiment analysis, and trading recommendations using NLP.

Designed as a data science portfolio project demonstrating applied NLP, financial domain knowledge, and production-style Python tooling.

---

## Overview

The Earnings Call Analyzer processes raw earnings call transcripts and automatically extracts:

- Revenue and year-over-year growth
- Margin indicators
- Forward guidance direction
- Management sentiment
- Key growth drivers

Based on these signals, the tool produces:

- BUY / SELL / HOLD recommendation
- Suggested options strategy
- Confidence level
- Analyst-ready CSV output

The recommendation logic is informed by academic research showing that **forward guidance and earnings surprises are strong predictors of post-earnings stock movement**.

---

## Example Usage

```bash
python earnings_analyzer.py --path AMD_2017_Q4.txt --use_huggingface
```

### Example Output

```text
AMD Q4 2017 → BUY (High)

Revenue: $1.48B (+34% YoY)
Guidance: Maintained
Sentiment: Positive
Strategy: Buy calls
```

---

## Output

The tool generates a CSV file suitable for downstream analysis:

```csv
company,ticker,quarter,year,revenue,revenue_growth_yoy,guidance,sentiment,recommendation,confidence
AMD,AMD,Q4,2017,1.48,34.0,maintained,positive,BUY,High
```

---

## Methodology

1. Parses unstructured earnings transcripts
2. Applies FinBERT for financial sentiment classification
3. Extracts financial metrics using NLP patterns
4. Scores results using a rule-based decision framework
5. Outputs structured data and trade recommendations

Primary decision factors:

- Forward guidance direction
- Revenue growth magnitude
- Margin trends
- Management tone

---

## Tech Stack

- Python
- Hugging Face Transformers (FinBERT)
- PyTorch
- pandas
- NumPy

---

## Use Cases

- Earnings call screening for investment research
- NLP portfolio demonstration
- Feature extraction from unstructured financial text
- Backtesting research pipelines

---

## Data Source
Earnings call transcripts were sourced from the [Earning Call Transcripts](https://www.kaggle.com/datasets/ramssvimala/earning-call-transcripts) dataset on Kaggle
The dataset provides raw earnings call text across multiple companies and quarters, which is used as input to the NLP pipeline for metric extraction and sentiment analysis.

---

## Disclaimer

This project is for educational and research purposes only.  
Not financial advice.
