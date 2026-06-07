"""
train.py — Single entrypoint for the full training pipeline.

Run: python train.py
     python train.py --model xgboost
     python train.py --model all --refresh-data

LESSON FOR ANY PROJECT:
  Your entrypoint should read like a table of contents.
  Anyone should understand the entire pipeline in 60 seconds just
  by reading this file. All complexity is hidden inside src/.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader   import load_market_data
from src.data.labeler  import compute_hmm_inputs, fit_hmm, save_hmm_artifacts
from src.features.pipeline import build_features, select_features, save_features
from src.models.trainer    import run_experiment, save_best_model, get_model_registry
from src.models.transition_detector import (
    build_transition_features, TransitionDetector,
    save_transition_detector,
)
from src.evaluation.metrics import (
    compute_summary, compare_models, print_classification_report,
    plot_confusion_matrix, plot_wfv_metrics, plot_predictions,
    plot_transition_alert, plot_alert_effectiveness,
)
from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Market Regime Prediction — Training Pipeline")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + list(get_model_registry().keys()),
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of market data (ignore cache)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (faster runs during debugging)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MARKET REGIME PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load Data ────────────────────────────────────────────────
    logger.info("\n[STEP 1/5] Loading market data...")
    market_data = load_market_data(force_refresh=args.refresh_data)
    spy    = market_data["price"]
    vix    = market_data["vix"]
    usdinr = market_data.get("usdinr")   # None if not configured
    crude  = market_data.get("crude")    # None if not configured

    # ── Step 2: HMM Regime Labeling ──────────────────────────────────────
    logger.info("\n[STEP 2/5] Fitting HMM regime labeler...")
    hmm_inputs   = compute_hmm_inputs(spy)
    labeled_df, hmm_model, hmm_scaler, state_map = fit_hmm(hmm_inputs)
    save_hmm_artifacts(hmm_model, hmm_scaler, state_map)

    dist = labeled_df["regime_name"].value_counts()
    for regime, count in dist.items():
        logger.info(f"  {regime:8s}: {count:5d} days ({count/len(labeled_df)*100:.1f}%)")

    # ── Step 3: Feature Engineering ──────────────────────────────────────
    logger.info("\n[STEP 3/5] Building feature matrix...")
    feature_matrix, feature_cols = build_features(spy, vix, labeled_df, usdinr_df=usdinr, crude_df=crude)

    # Quick RF for feature importance (to guide selection)
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    X = feature_matrix[feature_cols].values
    y = feature_matrix["target_regime_code"].values.astype(int)
    quick_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    quick_rf.fit(X, y)
    importances = pd.Series(quick_rf.feature_importances_, index=feature_cols)

    selected = select_features(feature_matrix, feature_cols, importances)
    save_features(feature_matrix, selected)
    logger.info(f"  Final feature count: {len(selected)}")

    # ── Step 4: Main Model Training ──────────────────────────────────────
    logger.info("\n[STEP 4/6] Training main regime classifiers (walk-forward)...")

    models_to_run = (
        list(get_model_registry().keys())
        if args.model == "all"
        else [args.model]
    )

    summaries      = []
    results_map    = {}
    all_preds_map  = {}
    trained_models = {}

    for model_name in models_to_run:
        fold_results, all_preds, final_model, scaler, run_id = run_experiment(
            model_name=model_name,
            feature_matrix=feature_matrix,
            feature_cols=selected,
        )
        summary = compute_summary(fold_results, model_name)
        summaries.append(summary)
        results_map[model_name]    = fold_results
        all_preds_map[model_name]  = all_preds
        trained_models[model_name] = (final_model, scaler)

    # ── Step 4b: Transition Detector ─────────────────────────────────────
    # A dedicated binary model that answers: "Will the regime change soon?"
    # Separate from main model — optimizes for recall over accuracy.
    logger.info("\n[STEP 4b/6] Training transition detector...")
    X_tr, y_tr, tr_feat_cols = build_transition_features(feature_matrix, horizon=5)

    detector = TransitionDetector(alert_threshold=0.35, horizon=5)
    tr_fold_results, tr_all_preds = detector.run_wfv(X_tr, y_tr, tr_feat_cols)
    detector.fit_final(X_tr, y_tr, tr_feat_cols)
    save_transition_detector(detector, tr_feat_cols)

    # ── Step 5: Evaluation & Save ─────────────────────────────────────────
    logger.info("\n[STEP 5/6] Evaluating and saving best model...")

    comparison = compare_models(summaries)
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON — Mean Across All WFV Folds")
    logger.info("=" * 70)
    logger.info("\n" + comparison.to_string())

    comparison.to_csv(cfg.training.model_comparison_path)

    # Select best model by F1 macro
    best_name            = comparison["f1_macro"].idxmax()
    best_model, best_scaler = trained_models[best_name]
    best_metrics         = summaries[[s["model"] for s in summaries].index(best_name)]

    save_best_model(
        model=best_model,
        scaler=best_scaler,
        model_name=best_name,
        feature_cols=selected,
        wfv_metrics=best_metrics,
    )

    logger.info(f"\n  Best model : {best_name}")
    logger.info(f"  F1 Macro   : {best_metrics['f1_macro']}")
    logger.info(f"  Accuracy   : {best_metrics['accuracy']}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("\n[STEP 6/6] Generating evaluation plots...")

        # WFV comparison
        plot_wfv_metrics(results_map)

        # Per-model plots
        for model_name in models_to_run:
            if model_name == "naive_baseline":
                continue
            preds = all_preds_map[model_name]
            print_classification_report(preds, model_name)
            plot_confusion_matrix(preds, model_name)
            plot_predictions(preds, model_name)
            plot_transition_alert(preds, model_name)
            plot_alert_effectiveness(preds)

        logger.info("  All plots saved to plots/")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Best model saved → {cfg.deployment.model_bundle_path}")
    logger.info("Run `mlflow ui` to view experiment results")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()