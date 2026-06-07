"""
src/evaluation/metrics.py

Responsibility: All evaluation logic — metrics computation and visualization.
This module never trains models. It only consumes predictions.

LESSON FOR ANY PROJECT:
  Keep plotting and metrics completely separate from training.
  Your trainer shouldn't know what a confusion matrix is.
  Your plotter shouldn't know what XGBoost is.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)

REGIME_COLOR = {"Bear": "#e74c3c", "Sideways": "#f39c12", "Bull": "#2ecc71"}
REGIME_NAMES = {0: "Bear", 1: "Sideways", 2: "Bull"}


# ────────────────────────────────────────────────────────────────────────────
# METRICS
# ────────────────────────────────────────────────────────────────────────────

def compute_summary(fold_results: pd.DataFrame, model_name: str) -> dict:
    """Compute mean metrics across all WFV folds."""
    agg = fold_results.mean(numeric_only=True)
    total_folds = int(len(fold_results))

    def _valid_fold_count(support_col: str, metric_col: str) -> int:
        if support_col in fold_results.columns:
            return int((fold_results[support_col] > 0).sum())
        return int(fold_results[metric_col].notna().sum())

    bear_valid = _valid_fold_count("support_bear", "f1_bear")
    side_valid = _valid_fold_count("support_sideways", "f1_sideways")
    bull_valid = _valid_fold_count("support_bull", "f1_bull")

    def _display(metric_value: float, valid_folds: int) -> str:
        if valid_folds == 0 or pd.isna(metric_value):
            return "N/A"
        return f"{metric_value:.4f}"

    summary = {
        "model":          model_name,
        "accuracy":       round(agg["accuracy"],       4),
        "f1_macro":       round(agg["f1_macro"],       4),
        "f1_weighted":    round(agg["f1_weighted"],    4),
        "f1_bear":        round(agg["f1_bear"],        4),
        "f1_sideways":    round(agg["f1_sideways"],    4),
        "f1_bull":        round(agg["f1_bull"],        4),
        "transition_acc": round(agg["transition_acc"], 4),
        "f1_bear_display": _display(agg["f1_bear"], bear_valid),
        "f1_sideways_display": _display(agg["f1_sideways"], side_valid),
        "f1_bull_display": _display(agg["f1_bull"], bull_valid),
        "f1_bear_eval_folds": f"{bear_valid}/{total_folds}",
        "f1_sideways_eval_folds": f"{side_valid}/{total_folds}",
        "f1_bull_eval_folds": f"{bull_valid}/{total_folds}",
    }
    return summary


def compare_models(summaries: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of summary dicts.
    Returns DataFrame sorted by f1_macro descending.
    """
    df = pd.DataFrame(summaries).set_index("model")
    df = df.sort_values("f1_macro", ascending=False)
    return df


def print_classification_report(all_preds: pd.DataFrame, model_name: str) -> None:
    """Print sklearn classification report for all WFV test predictions."""
    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name} (All WFV Folds)")
    print(f"{'='*60}")
    print(classification_report(
        all_preds["y_true"],
        all_preds["y_pred"],
        target_names=["Bear", "Sideways", "Bull"],
    ))


# ────────────────────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(all_preds: pd.DataFrame, model_name: str, save_dir: str = "plots") -> plt.Figure:
    """Confusion matrix (raw counts + normalized) across all WFV folds."""
    cm      = confusion_matrix(all_preds["y_true"], all_preds["y_pred"], labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ["Bear", "Sideways", "Bull"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Confusion Matrix (All WFV Folds)", fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw Counts", "Row-Normalized (Recall)"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    Path(save_dir).mkdir(exist_ok=True)
    path = f"{save_dir}/cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    return fig


def plot_wfv_metrics(
    results_map: dict[str, pd.DataFrame],
    save_dir: str = "plots",
) -> plt.Figure:
    """Per-fold metric lines for all models."""
    metrics = [
        ("f1_macro",       "F1 Macro"),
        ("f1_bear",        "F1 — Bear"),
        ("f1_bull",        "F1 — Bull"),
        ("transition_acc", "Transition Accuracy"),
    ]
    styles = {
        "naive_baseline": ("gray",      "--", "x"),
        "random_forest":  ("steelblue", "-",  "o"),
        "xgboost":        ("#e74c3c",   "-",  "s"),
        "lightgbm":       ("#2ecc71",   "-",  "^"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Walk-Forward Validation — Per-Fold Metrics", fontsize=13, fontweight="bold")

    for ax, (metric, title) in zip(axes.flat, metrics):
        for model_name, res in results_map.items():
            color, ls, marker = styles.get(model_name, ("black", "-", "o"))
            ax.plot(res["fold"], res[metric], color=color, linestyle=ls,
                    marker=marker, markersize=5, linewidth=1.5, label=model_name)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Score")
        ax.set_xlabel("Test Year")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"src/{save_dir}/wfv_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    return fig


def plot_predictions(all_preds: pd.DataFrame, model_name: str, save_dir: str = "plots") -> plt.Figure:
    """True vs predicted regime overlay on price chart."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    fig.suptitle(f"{model_name} — Predicted vs True Regime", fontsize=13, fontweight="bold")

    for ax, (label, col) in zip(axes, [("True Regime", "y_true"), ("Predicted Regime", "y_pred")]):
        ax.plot(all_preds.index, all_preds["Close"], color="black", linewidth=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_ylabel("Price ($)")
        ax.set_title(label)

        _shade_regimes(ax, all_preds.index, all_preds[col].map(REGIME_NAMES))

        patches = [mpatches.Patch(color=REGIME_COLOR[r], alpha=0.5, label=r)
                   for r in ["Bull", "Sideways", "Bear"]]
        ax.legend(handles=patches, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = f"{save_dir}/predictions_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    return fig


def plot_transition_alert(all_preds: pd.DataFrame, model_name: str, save_dir: str = "plots") -> plt.Figure:
    """
    The transition alert chart — the answer to low transition accuracy.

    Shows:
    1. Price chart with regime shading
    2. Regime probability stacked area (Bull/Sideways/Bear)
    3. Transition alert signal (red bars when confidence < threshold)

    HOW TO READ THIS:
    - When the green (Bull) probability drops and orange/red rise = uncertainty
    - Red bars in panel 3 = model is below confidence threshold = watch for regime change
    - The longer the consecutive alert streak, the stronger the signal
    """
    prob_cols = ["prob_bear", "prob_sideways", "prob_bull"]
    has_proba = all(c in all_preds.columns for c in prob_cols)
    has_alert = "transition_alert" in all_preds.columns

    n_panels = 3 if (has_proba and has_alert) else 1
    heights  = [3, 2, 1.5] if n_panels == 3 else [1]

    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 4 * n_panels),
                              sharex=True, gridspec_kw={"height_ratios": heights})
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"{model_name} — Transition Alert System\n"
        f"(Alert threshold: {cfg.training.transition_alert_threshold*100:.0f}% confidence)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Price + Predicted Regime ───────────────────────────────
    ax1 = axes[0]
    ax1.plot(all_preds.index, all_preds["Close"], color="black", linewidth=0.9, zorder=5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Price ($)", fontsize=10)
    ax1.set_title("Price + Predicted Regime")
    _shade_regimes(ax1, all_preds.index, all_preds["pred_regime"])
    patches = [mpatches.Patch(color=REGIME_COLOR[r], alpha=0.5, label=r)
               for r in ["Bull", "Sideways", "Bear"]]
    ax1.legend(handles=patches, loc="upper left", fontsize=9)

    if has_proba:
        # ── Panel 2: Stacked Probability ────────────────────────────────
        ax2 = axes[1]
        ax2.stackplot(
            all_preds.index,
            all_preds["prob_bear"],
            all_preds["prob_sideways"],
            all_preds["prob_bull"],
            labels=["Bear", "Sideways", "Bull"],
            colors=["#e74c3c", "#f39c12", "#2ecc71"],
            alpha=0.85,
        )
        ax2.axhline(
            cfg.training.transition_alert_threshold,
            color="white", linestyle="--", linewidth=1.2,
            label=f"Alert threshold ({cfg.training.transition_alert_threshold*100:.0f}%)"
        )
        ax2.set_ylabel("Probability", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title("Regime Probabilities — Stacked Area")
        ax2.legend(loc="upper left", fontsize=8)

    if has_alert:
        # ── Panel 3: Transition Alert Bars ──────────────────────────────
        ax3 = axes[2]
        alert      = all_preds["transition_alert"]
        severity   = all_preds.get("alert_severity", alert.astype(float))
        consec     = all_preds.get("consecutive_alert_days", alert.astype(int))

        # Bar height = severity, color intensity = consecutive days
        colors = ["#c0392b" if c >= 3 else "#e74c3c" if c >= 1 else "none"
                  for c in consec]
        ax3.bar(all_preds.index, severity, color=colors, width=1, alpha=0.8)
        ax3.set_ylabel("Alert Severity", fontsize=10)
        ax3.set_ylim(0, cfg.training.transition_alert_threshold)
        ax3.set_title(
            "⚠️  Transition Alert  |  "
            "Light red = alert active  |  Dark red = 3+ consecutive days"
        )
        ax3.set_xlabel("Date")

        # Annotate alert streaks ≥ 3 consecutive days
        if "consecutive_alert_days" in all_preds.columns:
            streak_starts = all_preds[
                (all_preds["consecutive_alert_days"] == 3)
            ].index
            for dt in streak_starts[:10]:  # limit annotations
                ax3.annotate(
                    "⚠️", xy=(dt, severity.loc[dt]),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=8, color="#c0392b", ha="center"
                )

    plt.tight_layout()
    path = f"{save_dir}/transition_alert_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    return fig


def plot_alert_effectiveness(all_preds: pd.DataFrame, save_dir: str = "plots") -> plt.Figure:
    """
    Measure how effective the transition alert actually is.
    Shows: what % of true regime transitions were preceded by an alert?
    """
    if "transition_alert" not in all_preds.columns:
        logger.warning("No transition_alert column — skipping effectiveness plot")
        return None

    # Find true regime change days
    true_changes = (all_preds["y_true"] != all_preds["y_true"].shift(1))
    true_change_idx = all_preds.index[true_changes]

    # For each true change, was an alert fired in the prior 5 days?
    lookahead = 5
    alert_preceded = []
    for dt in true_change_idx:
        loc = all_preds.index.get_loc(dt)
        window_start = max(0, loc - lookahead)
        window = all_preds.iloc[window_start:loc]["transition_alert"]
        alert_preceded.append(int(window.sum() > 0))

    coverage = np.mean(alert_preceded) if alert_preceded else 0

    # False alert rate
    non_change_days  = ~true_changes
    false_alert_rate = all_preds.loc[non_change_days, "transition_alert"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transition Alert Effectiveness", fontsize=13, fontweight="bold")

    # Alert coverage of true transitions
    axes[0].bar(
        ["Preceded by Alert", "No Alert"],
        [coverage, 1 - coverage],
        color=["#2ecc71", "#e74c3c"], alpha=0.8
    )
    axes[0].set_title(f"True Transitions Preceded by Alert (±{lookahead} days)")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([coverage, 1 - coverage]):
        axes[0].text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")

    # Alert rate breakdown
    rates = {
        "Alert on\nTransition Days":     all_preds.loc[true_changes,  "transition_alert"].mean(),
        "Alert on\nNon-Transition Days": false_alert_rate,
    }
    axes[1].bar(rates.keys(), rates.values(), color=["#e74c3c", "#95a5a6"], alpha=0.8)
    axes[1].set_title("Alert Rate: Transition vs Non-Transition Days")
    axes[1].set_ylabel("Alert Rate")
    axes[1].set_ylim(0, 1)
    for i, (k, v) in enumerate(rates.items()):
        axes[1].text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")

    # Summary text
    fig.text(
        0.5, -0.02,
        f"Alert Coverage: {coverage*100:.1f}% of true transitions | "
        f"False Alert Rate: {false_alert_rate*100:.1f}% of non-transition days",
        ha="center", fontsize=10, style="italic"
    )

    plt.tight_layout()
    path = f"{save_dir}/alert_effectiveness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    return fig


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _shade_regimes(ax, dates, regime_series):
    """Helper: shade price chart background by regime."""
    prev, start = None, None
    for date, regime in zip(dates, regime_series):
        if regime != prev:
            if prev and start:
                ax.axvspan(start, date, alpha=0.22, color=REGIME_COLOR.get(prev, "gray"), zorder=1)
            start, prev = date, regime
    if prev and start:
        ax.axvspan(start, dates[-1], alpha=0.22, color=REGIME_COLOR.get(prev, "gray"), zorder=1)
