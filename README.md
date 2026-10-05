# E-Commerce Product Sales – Exploratory Data Analysis

F.Y. BCA Semester I – Division B | Group 14

Exploratory data analysis of the Kaggle *E-Commerce Product Sales* dataset
(1,000 products × 7 columns: product_id, product_name, category, price, units_sold, rating, in_stock).

## Files
- `EDA_Analysis.ipynb` – full analysis notebook (run top to bottom)
- `ecommerce_dataset.csv` – dataset
- `requirements.txt` – Python packages
- `charts/` – the 8 figures are saved here when the notebook runs

## How to run
```
pip install -r requirements.txt
jupyter notebook EDA_Analysis.ipynb
```
Keep `ecommerce_dataset.csv` in the same folder as the notebook.

## What the notebook does
1. Loads the data and checks structure, missing values, duplicates and invalid values
2. Fixes 4 missing `category` values using the product name
3. Derives `sales = price × units_sold` (the dataset has no revenue column)
4. Descriptive statistics, grouping/aggregation by category and product
5. Eight charts, correlation analysis and business insights
