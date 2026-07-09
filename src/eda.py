"""Reusable, publication-quality EDA plotting functions.

Shared by every per-disease notebook in `notebooks/` so that plot styling,
figure sizing, and save-to-disk behavior stay consistent without duplicating
matplotlib/seaborn boilerplate in each notebook. Every function both returns
a `matplotlib.figure.Figure` (for inline notebook display) and saves a PNG to
`reports/figures/` (for the README and reports).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.titleweight"] = "bold"


def _save(fig: plt.Figure, disease_key: str, name: str) -> None:
    path = FIGURES_DIR / f"{disease_key}_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    logger.info("Saved figure to %s", path)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a transposed `describe()` table including dtype and missing count."""
    desc = df.describe(include="all").T
    desc["missing"] = df.isna().sum()
    desc["dtype"] = df.dtypes.astype(str)
    return desc


def plot_missing_values(df: pd.DataFrame, disease_key: str) -> plt.Figure:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(missing) + 1)))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=13)
        ax.axis("off")
    else:
        pct = (missing / len(df) * 100).round(1)
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index, ax=ax, legend=False)
        for i, (count, p) in enumerate(zip(missing.values, pct.values)):
            ax.text(count, i, f"  {count} ({p}%)", va="center")
        ax.set_xlabel("Missing count")
    ax.set_title("Missing Values per Column (raw data, before imputation)")
    fig.tight_layout()
    _save(fig, disease_key, "missing_values")
    return fig


def plot_target_prevalence(
    df: pd.DataFrame, target_col: str, labels: dict[int, str], disease_key: str
) -> plt.Figure:
    counts = df[target_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [labels.get(i, str(i)) for i in counts.index],
        counts.values,
        color=sns.color_palette("viridis", len(counts)),
    )
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({count/total:.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Number of patients")
    ax.set_title("Disease Prevalence in Dataset")
    fig.tight_layout()
    _save(fig, disease_key, "prevalence")
    return fig


def plot_numeric_distributions(
    df: pd.DataFrame, columns: list[str], target_col: str, disease_key: str
) -> plt.Figure:
    n = len(columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(columns):
        sns.histplot(
            data=df,
            x=col,
            hue=target_col,
            kde=True,
            ax=axes[i],
            element="step",
            stat="density",
            common_norm=False,
        )
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions by Disease Status", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, disease_key, "distributions")
    return fig


def plot_boxplots(df: pd.DataFrame, columns: list[str], target_col: str, disease_key: str) -> plt.Figure:
    n = len(columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(columns):
        sns.boxplot(data=df, x=target_col, y=col, hue=target_col, ax=axes[i], legend=False)
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Boxplots by Disease Status (outlier visualization)", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, disease_key, "boxplots")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, disease_key: str) -> plt.Figure:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(0.55 * len(corr.columns) + 3, 0.55 * len(corr.columns) + 2))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    _save(fig, disease_key, "correlation_heatmap")
    return fig


def plot_pairplot(
    df: pd.DataFrame, columns: list[str], target_col: str, disease_key: str, labels: dict[int, str]
):
    plot_df = df[columns + [target_col]].copy()
    plot_df[target_col] = plot_df[target_col].map(labels)
    grid = sns.pairplot(
        plot_df, hue=target_col, diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 20}
    )
    grid.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14)
    grid.fig.tight_layout()
    path = FIGURES_DIR / f"{disease_key}_pairplot.png"
    grid.fig.savefig(path, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    return grid.fig


def target_correlation_ranked(df: pd.DataFrame, target_col: str) -> pd.Series:
    return df.corr(numeric_only=True)[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)
