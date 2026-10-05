import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("charts", exist_ok=True)
pd.options.display.float_format = "{:,.2f}".format
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED = "#2E75B6", "#ED7D31", "#70AD47", "#C00000"
FIGSIZE = (7.9, 4.9)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"charts/{name}.png", dpi=200)
    plt.show()

df = pd.read_csv("ecommerce_dataset.csv")
print("Shape (rows, columns):", df.shape)
df.head()

df.info()

print("Missing values per column:")
print(df.isna().sum(), "\n")
print("Duplicate rows (ignoring product_id):", df.drop(columns="product_id").duplicated().sum())
print("Duplicate product_id values:", df["product_id"].duplicated().sum())
print("Price <= 0:", (df["price"] <= 0).sum(), "| Units sold <= 0:", (df["units_sold"] <= 0).sum())
print("Ratings outside 1-5:", (~df["rating"].between(1, 5)).sum())
print("\nRows with a missing category:")
df[df["category"].isna()]

# Every product name belongs to exactly one category in the complete rows,
# so the 4 missing categories can be filled from the product name.
known = df.dropna(subset=["category"])
assert known.groupby("product_name")["category"].nunique().max() == 1
mapping = known.drop_duplicates("product_name").set_index("product_name")["category"]
df["category"] = df["category"].fillna(df["product_name"].map(mapping))

print("Missing values after cleaning:", int(df.isna().sum().sum()))
print(df["category"].value_counts())

# The dataset has no sales/revenue column, so derive it.
df["sales"] = df["price"] * df["units_sold"]
print("Total derived sales: ₹{:,.2f}".format(df["sales"].sum()))

cols = ["price", "units_sold", "rating", "sales"]
stats = df[cols].agg(["count", "mean", "median", "std", "min", "max"]).T
stats["skewness"] = df[cols].skew()
stats

by_category = df.groupby("category").agg(
    records=("sales", "size"), total_sales=("sales", "sum"), avg_sales=("sales", "mean"),
    units_sold=("units_sold", "sum"), avg_rating=("rating", "mean"),
    avg_price=("price", "mean"), in_stock=("in_stock", "sum"))
by_category["sales_share_%"] = by_category["total_sales"] / df["sales"].sum() * 100
by_category

by_product = df.groupby("product_name").agg(
    records=("sales", "size"), total_sales=("sales", "sum"), avg_sales=("sales", "mean"),
    avg_rating=("rating", "mean"), avg_price=("price", "mean")).sort_values("total_sales", ascending=False)
by_product

# Figure 1 - total derived sales by category
c = df.groupby("category")["sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(c.index, c.values / 1e6, color=[BLUE, ORANGE], width=0.5)
for b, v in zip(bars, c.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"₹{v/1e6:,.2f}M", ha="center", fontweight="bold")
ax.set(xlabel="Category", ylabel="Total derived sales (₹millions)", ylim=(0, c.max() / 1e6 * 1.15))
save(fig, "fig1_sales_by_category")

# Figure 2 - total units sold by category
u = df.groupby("category")["units_sold"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(u.index, u.values, color=[BLUE, ORANGE], width=0.5)
for b, v in zip(bars, u.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1500, f"{v:,}", ha="center", fontweight="bold")
ax.set(xlabel="Category", ylabel="Total units sold", ylim=(0, u.max() * 1.15))
save(fig, "fig2_units_by_category")

# Figure 3 - total derived sales by product
p = df.groupby("product_name")["sales"].sum().sort_values()
fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.barh(p.index, p.values / 1e6, color=BLUE)
for b, v in zip(bars, p.values):
    ax.text(v / 1e6 + 0.2, b.get_y() + b.get_height() / 2, f"₹{v/1e6:,.2f}M", va="center", fontsize=11)
ax.set(xlabel="Total derived sales (₹millions)", xlim=(0, p.max() / 1e6 * 1.2))
save(fig, "fig3_sales_by_product")

# Figure 4 - stock status by category
s = df.groupby(["category", "in_stock"]).size().unstack()
s = s.rename(columns={True: "In stock", False: "Out of stock"})[["In stock", "Out of stock"]]
x, w = np.arange(len(s)), 0.35
fig, ax = plt.subplots(figsize=FIGSIZE)
b1 = ax.bar(x - w / 2, s["In stock"], w, color=GREEN, label="In stock")
b2 = ax.bar(x + w / 2, s["Out of stock"], w, color=RED, label="Out of stock")
for group in (b1, b2):
    for b in group:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, int(b.get_height()), ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(s.index)
ax.set(xlabel="Category", ylabel="Number of products", ylim=(0, s.values.max() * 1.15))
ax.legend(frameon=False)
save(fig, "fig4_stock_by_category")

# Figure 5 - average rating by product
r = df.groupby("product_name")["rating"].mean().sort_values()
fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.barh(r.index, r.values, color=ORANGE)
for b, v in zip(bars, r.values):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=11)
ax.axvline(df["rating"].mean(), color="black", ls="--", lw=1)
ax.set(xlabel="Average rating (1-5)", xlim=(0, 5))
save(fig, "fig5_rating_by_product")

# Figure 6 - distribution of derived sales
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.hist(df["sales"] / 1e3, bins=30, color=BLUE, edgecolor="white")
ax.axvline(df["sales"].mean() / 1e3, color=RED, ls="--", label=f"Mean ₹{df['sales'].mean()/1e3:,.1f}K")
ax.axvline(df["sales"].median() / 1e3, color="black", ls=":", label=f"Median ₹{df['sales'].median()/1e3:,.1f}K")
ax.set(xlabel="Derived sales per product record (₹thousands)", ylabel="Frequency")
ax.legend(frameon=False)
save(fig, "fig6_sales_distribution")

# Figure 7 - price vs units sold
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(df["price"], df["units_sold"], s=14, alpha=0.55, color="#1B9E9E")
ax.set(xlabel="Price (Rs)", ylabel="Units sold")
save(fig, "fig7_price_vs_units")

# product_id (identifier) and sales (derived from price x units) are excluded.
corr = df[["price", "units_sold", "rating"]].corr()
corr.columns = corr.index = ["Price", "Units Sold", "Rating"]
print(corr.round(4))

# Figure 8 - correlation heatmap
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.heatmap(corr.round(2), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            annot_kws={"size": 15}, cbar_kws={"label": "Pearson correlation"}, ax=ax)
save(fig, "fig8_correlation_heatmap")

total = df["sales"].sum()
E, F = by_category.loc["Electronics"], by_category.loc["Fashion"]
print(f"Electronics: ₹{E.total_sales:,.2f} ({E.total_sales/total:.1%}) | Fashion: ₹{F.total_sales:,.2f} ({F.total_sales/total:.1%})")
print(f"Average sales per record: Electronics ₹{E.avg_sales:,.0f} vs Fashion ₹{F.avg_sales:,.0f}")
print("Top product:", by_product.index[0], f"₹{by_product.total_sales.iloc[0]:,.2f}",
      "| Lowest:", by_product.index[-1], f"₹{by_product.total_sales.iloc[-1]:,.2f}")
print("Best average rating:", by_product.avg_rating.idxmax(), round(by_product.avg_rating.max(), 2),
      "| Lowest:", by_product.avg_rating.idxmin(), round(by_product.avg_rating.min(), 2),
      "| Overall mean:", round(df.rating.mean(), 2))

out = df[~df["in_stock"]]
print(f"\nOut of stock: {len(out)} products ({len(out)/len(df):.1%}), derived sales ₹{out.sales.sum():,.2f} ({out.sales.sum()/total:.1%})")

high_price = df["price"] >= df["price"].median()
print("\nAbove vs below median price (avg units sold, avg sales per record):")
print(df.groupby(high_price).agg(units=("units_sold", "mean"), sales=("sales", "mean")).rename(index={True: "Above median", False: "Below median"}))

top10 = df.nlargest(len(df) // 10, "sales")["sales"].sum() / total
print(f"\nTop 10% of records produce {top10:.1%} of total derived sales")
print(f"Ratings >= 4: {(df.rating >= 4).mean():.1%} | Ratings < 2: {(df.rating < 2).mean():.1%}")