# Sales Data Analysis

An exploratory data analysis project using Python to uncover patterns, trends, customer behavior, and business insights from transactional sales data.

## Project Overview

This project demonstrates a complete exploratory data analysis workflow, from data cleaning and validation to statistical analysis, visualization, customer analysis, and correlation analysis.

## Key Results

| Metric | Result |
|---|---:|
| Total Sales | 15,547.35 |
| Transactions Analyzed | 277 |
| Customers Represented | 86 |
| Average Transaction Value | 56.13 |
| Highest Daily Sales | 1,147.62 |
| Highest-Value Customer | Customer 48 |
| Sales–Transaction Correlation | 0.694 |

## Analysis Performed

- Data inspection and validation
- Missing-value analysis
- Duplicate detection
- Date conversion and preprocessing
- Descriptive statistics
- Sales distribution analysis
- Outlier detection using IQR
- Daily sales trend analysis
- Customer-level sales analysis
- Transaction-volume analysis
- Correlation analysis

## Key Insights

- The sales distribution is right-skewed, with the mean (56.13) higher than the median (36.28).
- Six high-value transactions were identified as statistical outliers and retained as valid observations.
- September 22 recorded the highest daily sales at 1,147.62.
- September 9 recorded the lowest daily sales at 119.70.
- Customer 48 generated the highest total sales at 524.41 across four transactions.
- Daily transaction volume and total daily sales showed a correlation of 0.694, indicating a moderately strong positive relationship.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab


## Project Structure

```text
sales-data-analysis/
├── notebooks/
│   └── 01_sales_eda.ipynb
├── 01_sales_data_analysis.py
├── grocery_sales.csv
└── README.md
```
## How to Use

1. Clone or download this repository.
2. Open `notebooks/01_sales_eda.ipynb` in Jupyter Notebook or Google Colab.
3. Ensure `grocery_sales.csv` is available in the notebook environment.
4. Run the notebook cells sequentially.

## Future Development

Future versions may extend the analysis with:

- Interactive dashboards
- Customer segmentation
- Predictive sales modelling
- Machine learning
- Advanced statistical analysis
### Author

Dr. Hira Benish
Associate Professor of Mathematics | Data Science | Deep Learning | Artificial Intelligence

Riphah International University, Sahiwal
