# Retail Sales Exploratory Analysis

An evidence-led exploratory data analysis of one month of transaction data. The project demonstrates a reproducible workflow for validating data, examining revenue patterns, understanding customer behaviour, and translating statistical findings into appropriately qualified business insights.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrHiraBenish/sales-data-analysis/blob/main/notebooks/01_sales_eda.ipynb)

## Project at a Glance

| Metric | Result |
|---|---:|
| Raw records | 281 |
| Records analysed | 277 |
| Missing sales values | 4 (1.4%) |
| Total recorded sales | 15,547.35 |
| Unique customers | 86 |
| Average transaction | 56.13 |
| Median transaction | 36.28 |
| Highest-sales day | 22 September 2020 (1,147.62) |

![Retail sales analysis dashboard](reports/figures/overview_dashboard.svg)

## Questions Addressed

1. Is the dataset complete, internally consistent, and suitable for analysis?
2. What do the centre, spread, and shape of transaction values reveal?
3. How do total sales and transaction volume change across the month?
4. Which customers contribute the most recorded sales?
5. Is daily sales performance associated with transaction volume?
6. Which conclusions are supported by this dataset—and which are not?

## Analytical Approach

```text
Raw data
   ↓
Validation and quality checks
   ↓
Transparent missing-value treatment
   ↓
Transaction, daily, and customer-level summaries
   ↓
Distribution and outlier analysis
   ↓
Relationship analysis and qualified interpretation
   ↓
Reproducible tables, figures, and conclusions
```

Four records have no sales amount. They are excluded from revenue calculations rather than mean-imputed because imputation would create artificial revenue. The original file remains unchanged, and the exclusion is reported explicitly.

## Key Findings

- Recorded sales total **15,547.35** across **277 complete transactions** from **86 customers**.
- The distribution is right-skewed: the mean transaction (**56.13**) is substantially above the median (**36.28**). A typical transaction is therefore represented more reliably by the median than by the mean alone.
- Six high-value observations exceed the IQR upper fence of **185.36**. They account for **7.6%** of recorded sales and are retained because the available fields provide no evidence that they are data-entry errors.
- **22 September 2020** produced the highest daily sales (**1,147.62**); the lowest was **9 September** (**119.70**). With only one month of data, these are descriptive observations rather than evidence of seasonality.
- Customer 48 generated the highest recorded sales (**524.41** across four transactions). The top 10 customers contributed **26.3%** of total sales, suggesting that revenue is not dominated by a single customer.
- **74.4%** of observed customers made more than one transaction and contributed **87.0%** of recorded sales. This is within-month repeat activity, not a long-term retention rate.
- Daily transaction count and daily sales have a Pearson correlation of **0.694**. This relationship is expected in part because daily sales is mathematically the sum of transaction values; it should not be presented as causal evidence.

## Reproducibility

```bash
git clone https://github.com/DrHiraBenish/sales-data-analysis.git
cd sales-data-analysis
python -m venv .venv
```

Activate the environment, then install the dependencies and run the analysis:

```bash
pip install -r requirements.txt
python src/analyze_sales.py
```

The script reads `data/grocery_sales.csv` and regenerates all tables and figures in `reports/`.

## Repository Structure

```text
sales-data-analysis/
├── data/
│   └── grocery_sales.csv
├── notebooks/
│   └── 01_sales_eda.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── analyze_sales.py
├── requirements.txt
└── README.md
```

## Data Dictionary

| Field | Type | Description |
|---|---|---|
| `customer_id` | Integer | Anonymised customer identifier |
| `transaction_date` | Date | Date of the recorded transaction |
| `transaction_id` | Integer | Unique transaction identifier |
| `sales` | Numeric | Recorded transaction sales amount; currency is not specified |

## Limitations

- The dataset covers only **September 2020**, so it cannot support seasonal or long-term forecasting claims.
- Product, category, location, cost, and profit fields are unavailable; product performance and profitability cannot be evaluated.
- Customer identifiers show activity within the observed month only; they do not establish acquisition, churn, or long-term retention.
- The original source, sampling method, and currency were not documented in the earlier repository. Results should therefore be treated as a portfolio demonstration, not as generalisable retail evidence.
- The small sample means that segment comparisons should remain descriptive.

## Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook · Google Colab

## Author

**Dr. Hira Benish** — Associate Professor of Mathematics · Researcher · Mathematics Behind AI
