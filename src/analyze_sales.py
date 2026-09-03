"""Reproducible exploratory analysis for the retail sales portfolio project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "grocery_sales.csv"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
EXPECTED_COLUMNS = {
    "customer_id",
    "transaction_date",
    "transaction_id",
    "sales",
}

NAVY = "#183B56"
BLUE = "#2F80ED"
TEAL = "#19A7A0"
VIOLET = "#7257D5"
ORANGE = "#F2994A"
LIGHT = "#EAF2F8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, analyse, and visualise the retail sales dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the source CSV and enforce the assumptions used by the analysis."""
    df = pd.read_csv(path)
    missing_columns = EXPECTED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.loc[:, sorted(EXPECTED_COLUMNS)].copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="raise")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique.")
    if df["customer_id"].isna().any():
        raise ValueError("Customer IDs contain missing values.")
    if (df["sales"].dropna() <= 0).any():
        raise ValueError("Sales values must be positive when present.")

    return df.sort_values(["transaction_date", "transaction_id"]).reset_index(drop=True)


def prepare_analysis_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a complete-case dataset plus daily and customer summaries."""
    clean = df.dropna(subset=["sales"]).copy()
    clean["day_name"] = clean["transaction_date"].dt.day_name()
    clean["day_type"] = np.where(
        clean["transaction_date"].dt.dayofweek >= 5, "Weekend", "Weekday"
    )

    daily = (
        clean.groupby("transaction_date", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_transaction=("sales", "mean"),
            transactions=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .sort_values("transaction_date")
    )

    customer = (
        clean.groupby("customer_id", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_transaction=("sales", "mean"),
            transactions=("transaction_id", "nunique"),
            first_purchase=("transaction_date", "min"),
            last_purchase=("transaction_date", "max"),
        )
        .sort_values("total_sales", ascending=False)
    )
    customer["activity_group"] = np.where(
        customer["transactions"] > 1, "Repeat within month", "One transaction"
    )

    return clean, daily, customer


def bootstrap_interval(
    values: pd.Series, statistic: str, seed: int = 42, samples: int = 10_000
) -> list[float]:
    """Return a reproducible percentile bootstrap interval."""
    array = values.to_numpy()
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))]
    estimates = draws.mean(axis=1) if statistic == "mean" else np.median(draws, axis=1)
    return [round(value, 2) for value in np.quantile(estimates, [0.025, 0.975])]


def calculate_metrics(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    daily: pd.DataFrame,
    customer: pd.DataFrame,
) -> dict[str, object]:
    q1, q3 = clean["sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outliers = clean.loc[clean["sales"] > upper_fence]
    repeat_customers = customer.loc[customer["transactions"] > 1]
    top_10 = customer.head(10)
    high_day = daily.loc[daily["total_sales"].idxmax()]
    low_day = daily.loc[daily["total_sales"].idxmin()]

    return {
        "raw_records": int(len(raw)),
        "records_analyzed": int(len(clean)),
        "missing_sales": int(raw["sales"].isna().sum()),
        "missing_sales_percent": round(raw["sales"].isna().mean() * 100, 1),
        "date_start": clean["transaction_date"].min().date().isoformat(),
        "date_end": clean["transaction_date"].max().date().isoformat(),
        "total_sales": round(float(clean["sales"].sum()), 2),
        "unique_customers": int(clean["customer_id"].nunique()),
        "mean_transaction": round(float(clean["sales"].mean()), 2),
        "median_transaction": round(float(clean["sales"].median()), 2),
        "mean_95_percent_bootstrap_interval": bootstrap_interval(
            clean["sales"], "mean"
        ),
        "median_95_percent_bootstrap_interval": bootstrap_interval(
            clean["sales"], "median"
        ),
        "iqr_upper_fence": round(float(upper_fence), 2),
        "high_value_transactions": int(len(outliers)),
        "high_value_sales_share_percent": round(
            float(outliers["sales"].sum() / clean["sales"].sum() * 100), 1
        ),
        "highest_sales_day": high_day["transaction_date"].date().isoformat(),
        "highest_daily_sales": round(float(high_day["total_sales"]), 2),
        "lowest_sales_day": low_day["transaction_date"].date().isoformat(),
        "lowest_daily_sales": round(float(low_day["total_sales"]), 2),
        "top_customer_id": int(customer.iloc[0]["customer_id"]),
        "top_customer_sales": round(float(customer.iloc[0]["total_sales"]), 2),
        "top_customer_transactions": int(customer.iloc[0]["transactions"]),
        "top_10_sales_share_percent": round(
            float(top_10["total_sales"].sum() / clean["sales"].sum() * 100), 1
        ),
        "repeat_customer_share_percent": round(
            float(len(repeat_customers) / len(customer) * 100), 1
        ),
        "repeat_customer_sales_share_percent": round(
            float(repeat_customers["total_sales"].sum() / clean["sales"].sum() * 100),
            1,
        ),
        "daily_sales_transaction_correlation": round(
            float(daily["total_sales"].corr(daily["transactions"])), 3
        ),
    }


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.edgecolor": "#C9D6E2",
            "grid.color": "#E6EEF5",
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    clean: pd.DataFrame,
    daily: pd.DataFrame,
    customer: pd.DataFrame,
    metrics: dict[str, object],
    output: Path,
) -> None:
    configure_plot_style()
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [4, 1]}
    )
    sns.histplot(clean, x="sales", bins=20, kde=True, color=BLUE, ax=axes[0])
    axes[0].axvline(clean["sales"].median(), color=TEAL, linestyle="--", label="Median")
    axes[0].axvline(clean["sales"].mean(), color=ORANGE, linestyle="--", label="Mean")
    axes[0].set(title="Transaction-value distribution", xlabel="Sales amount")
    axes[0].legend(frameon=False)
    sns.boxplot(data=clean, x="sales", color=LIGHT, linecolor=NAVY, ax=axes[1])
    axes[1].set(xlabel="Sales amount", ylabel="")
    save_figure(fig, output, "sales_distribution")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(daily["transaction_date"], daily["total_sales"], color=BLUE, marker="o")
    high = daily.loc[daily["total_sales"].idxmax()]
    low = daily.loc[daily["total_sales"].idxmin()]
    ax.scatter(
        [high["transaction_date"], low["transaction_date"]],
        [high["total_sales"], low["total_sales"]],
        color=[TEAL, ORANGE],
        s=75,
        zorder=3,
    )
    ax.annotate(
        f"High: {high['total_sales']:,.2f}",
        (high["transaction_date"], high["total_sales"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate(
        f"Low: {low['total_sales']:,.2f}",
        (low["transaction_date"], low["total_sales"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set(
        title="Daily recorded sales — September 2020",
        xlabel="Date",
        ylabel="Total sales",
    )
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, output, "daily_sales")

    top_10 = customer.head(10).sort_values("total_sales")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.barplot(
        top_10,
        x="total_sales",
        y=top_10["customer_id"].astype(str),
        color=TEAL,
        ax=ax,
    )
    ax.set(
        title="Top 10 customers by recorded sales",
        xlabel="Total sales",
        ylabel="Customer ID",
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=4, color=NAVY)
    save_figure(fig, output, "top_customers")

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sns.regplot(
        data=daily,
        x="transactions",
        y="total_sales",
        scatter_kws={"s": 55, "alpha": 0.8, "color": VIOLET},
        line_kws={"color": ORANGE},
        ax=ax,
    )
    ax.text(
        0.04,
        0.93,
        f"Pearson r = {metrics['daily_sales_transaction_correlation']}",
        transform=ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D5E1EC",
        },
    )
    ax.set(
        title="Daily sales and transaction volume",
        xlabel="Transactions per day",
        ylabel="Total daily sales",
    )
    save_figure(fig, output, "sales_vs_volume")

    activity = (
        customer.groupby("activity_group", as_index=False)
        .agg(customers=("customer_id", "nunique"), total_sales=("total_sales", "sum"))
        .sort_values("customers")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    sns.barplot(
        activity,
        x="activity_group",
        y="customers",
        hue="activity_group",
        palette=[ORANGE, TEAL],
        legend=False,
        ax=axes[0],
    )
    sns.barplot(
        activity,
        x="activity_group",
        y="total_sales",
        hue="activity_group",
        palette=[ORANGE, TEAL],
        legend=False,
        ax=axes[1],
    )
    axes[0].set(title="Observed customers", xlabel="", ylabel="Count")
    axes[1].set(title="Recorded sales contribution", xlabel="", ylabel="Total sales")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
    save_figure(fig, output, "customer_activity")

    create_overview_dashboard(clean, daily, customer, metrics, output)


def create_overview_dashboard(
    clean: pd.DataFrame,
    daily: pd.DataFrame,
    customer: pd.DataFrame,
    metrics: dict[str, object],
    output: Path,
) -> None:
    """Create the compact figure displayed in the repository README."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    sns.histplot(clean, x="sales", bins=20, color=BLUE, ax=axes[0, 0])
    axes[0, 0].axvline(
        clean["sales"].median(), color=TEAL, linestyle="--", label="Median"
    )
    axes[0, 0].axvline(
        clean["sales"].mean(), color=ORANGE, linestyle="--", label="Mean"
    )
    axes[0, 0].set(title="Transaction-value distribution", xlabel="Sales amount")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        daily["transaction_date"],
        daily["total_sales"],
        color=BLUE,
        marker="o",
        markersize=4,
    )
    axes[0, 1].set(title="Daily recorded sales", xlabel="Date", ylabel="Total sales")
    axes[0, 1].tick_params(axis="x", rotation=35)

    top_10 = customer.head(10).sort_values("total_sales")
    sns.barplot(
        top_10,
        x="total_sales",
        y=top_10["customer_id"].astype(str),
        color=TEAL,
        ax=axes[1, 0],
    )
    axes[1, 0].set(title="Top 10 customers", xlabel="Total sales", ylabel="Customer ID")

    sns.regplot(
        daily,
        x="transactions",
        y="total_sales",
        scatter_kws={"s": 42, "alpha": 0.8, "color": VIOLET},
        line_kws={"color": ORANGE},
        ax=axes[1, 1],
    )
    axes[1, 1].set(
        title=(
            "Sales vs volume "
            f"(r = {metrics['daily_sales_transaction_correlation']})"
        ),
        xlabel="Transactions per day",
        ylabel="Total daily sales",
    )

    fig.suptitle(
        "Retail Sales Exploratory Analysis",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
        y=1.01,
    )
    save_figure(fig, output, "overview_dashboard")


def write_outputs(
    raw: pd.DataFrame,
    daily: pd.DataFrame,
    customer: pd.DataFrame,
    metrics: dict[str, object],
    output: Path,
) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    daily.to_csv(tables / "daily_sales_summary.csv", index=False)
    customer.to_csv(tables / "customer_summary.csv", index=False)
    raw.loc[raw["sales"].isna()].to_csv(
        tables / "missing_sales_records.csv", index=False
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def print_summary(metrics: dict[str, object]) -> None:
    print("Retail Sales Exploratory Analysis")
    print("-" * 35)
    print(f"Records analysed: {metrics['records_analyzed']:,}")
    print(f"Total recorded sales: {metrics['total_sales']:,.2f}")
    print(f"Unique customers: {metrics['unique_customers']:,}")
    print(f"Mean transaction: {metrics['mean_transaction']:,.2f}")
    print(f"Median transaction: {metrics['median_transaction']:,.2f}")
    print(f"Missing sales values: {metrics['missing_sales']}")
    print("Reports regenerated successfully.")


def main() -> None:
    args = parse_args()
    raw = load_and_validate(args.data)
    clean, daily, customer = prepare_analysis_tables(raw)
    metrics = calculate_metrics(raw, clean, daily, customer)
    create_figures(clean, daily, customer, metrics, args.output / "figures")
    write_outputs(raw, daily, customer, metrics, args.output)
    print_summary(metrics)


if __name__ == "__main__":
    main()
