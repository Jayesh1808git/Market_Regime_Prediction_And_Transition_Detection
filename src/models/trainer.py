"""
src/models/trainer.py

Responsibility: Walk-forward training, model registry, transition alert logic.

KEY CONCEPT — TRANSITION ALERT:
  The model outputs 3 probabilities: P(Bear), P(Sideways), P(Bull).
  When the top-class probability drops below the alert threshold (default 0.60),
  the model is uncertain — this IS the regime change signal.
  We surface this as a 'transition_alert' flag in predictions.

LESSON FOR ANY PROJECT:
  Wrap your model training in a class that separates:
  - fit() — training
  - predict() — inference
  - predict_proba() — probability output (always implement this)
  This makes it trivial to swap models without changing downstream code.
"""

import numpy as np
import pandas as pd
import pickle
import inspect
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)

REGIME_NAMES  = {0: "Bear", 1: "Sideways", 2: "Bull"}


def _class_f1_if_present(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> float:
    """
    Return per-class F1 only when that class exists in y_true for the fold.
    If class is absent in test labels, return NaN instead of an artificial 0.
    """
    if not np.any(y_true == class_id):
        return np.nan
    return f1_score((y_true == class_id), (y_pred == class_id), zero_division=0)


def _fit_with_balanced_weights(model, X_tr_df: pd.DataFrame, y_tr: np.ndarray, model_name: str):
    """
    Fit model with per-sample balanced class weights.
    XGBoost requires 0-based contiguous labels — remaps if fold is missing a class.
    """
    if model_name == "naive_baseline":
        model.fit(X_tr_df, y_tr)
        return

    # Remap to 0-based contiguous integers (fixes XGBoost crash on [1,2] folds)
    unique_classes = np.unique(y_tr)
    label_map      = {orig: idx for idx, orig in enumerate(unique_classes)}
    reverse_map    = {idx: orig for orig, idx in label_map.items()}
    needs_remap    = list(unique_classes) != list(range(len(unique_classes)))

    y_tr_fit = np.array([label_map[c] for c in y_tr]) if needs_remap else y_tr

    # Store on model so the predict step can reverse
    model._label_remap         = label_map
    model._label_remap_reverse = reverse_map
    model._label_needs_remap   = needs_remap
    model._label_classes       = unique_classes

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr_fit)
    try:
        model.fit(X_tr_df, y_tr_fit, sample_weight=sample_weight)
    except TypeError:
        model.fit(X_tr_df, y_tr_fit)


# ────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Add a new model here and it automatically works in the full pipeline.
# LESSON: This pattern is called a "factory" — a dict that maps names to
# constructor functions. Never use if/elif chains to select models.
# ────────────────────────────────────────────────────────────────────────────

def get_model_registry() -> dict:
    """
    Returns a dict mapping model name → (constructor_fn, needs_scaling, mlflow_logger).
    Add new models here — the rest of the pipeline picks them up automatically.
    """
    rc  = cfg.training.models.random_forest
    xc  = cfg.training.models.xgboost
    lc  = cfg.training.models.lightgbm
    seed = cfg.training.random_seed

    # Registry format: { name: (model_factory, needs_scaling) }
    # mlflow logging is now handled centrally by _log_model_safely()
    return {
        "naive_baseline": (
            lambda: DummyClassifier(strategy="most_frequent", random_state=seed),
            False,
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=rc.n_estimators,
                max_depth=rc.max_depth,
                min_samples_leaf=rc.min_samples_leaf,
                max_features=rc.max_features,
                class_weight=rc.class_weight,   # "balanced_subsample" in config
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        "xgboost": (
            lambda: xgb.XGBClassifier(
                n_estimators=xc.n_estimators,
                max_depth=xc.max_depth,
                learning_rate=xc.learning_rate,
                subsample=xc.subsample,
                colsample_bytree=xc.colsample_bytree,
                reg_alpha=xc.reg_alpha,
                reg_lambda=xc.reg_lambda,
                min_child_weight=xc.get("min_child_weight", 1),
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
            ),
            True,
        ),
        "lightgbm": (
            lambda: lgb.LGBMClassifier(
                n_estimators=lc.n_estimators,
                max_depth=lc.max_depth,
                learning_rate=lc.learning_rate,
                num_leaves=lc.num_leaves,
                subsample=lc.subsample,
                colsample_bytree=lc.colsample_bytree,
                reg_alpha=lc.reg_alpha,
                reg_lambda=lc.reg_lambda,
                class_weight=lc.class_weight,
                is_unbalance=lc.get("is_unbalance", False),
                objective="multiclass",
                num_class=3,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            True,
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATOR
# ────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Expanding-window walk-forward validator.
    The core evaluation engine — handles all fold logic, scaling, and metric collection.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "target_regime_code",
    ):
        self.df           = df
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.start_year   = cfg.training.wfv_start_year
        self.end_year     = cfg.training.wfv_end_year
        self.alert_thresh = cfg.training.transition_alert_threshold

    def _folds(self):
        for year in range(self.start_year, self.end_year + 1):
            train_mask = self.df.index.year < year
            test_mask  = self.df.index.year == year
            if train_mask.sum() < 252 * 5 or test_mask.sum() < 50:
                continue
            yield train_mask, test_mask, year

    def run(
        self,
        model_factory,
        needs_scaling: bool = True,
        model_name: str = "model",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run walk-forward validation.

        Returns:
            fold_results : DataFrame of per-fold metrics
            all_preds    : DataFrame of all predictions with transition alerts
        """
        fold_results   = []
        all_preds_list = []

        for train_mask, test_mask, year in self._folds():

            # Keep as DataFrames — LightGBM needs feature names to match fit() time
            X_tr_df = self.df.loc[train_mask, self.feature_cols]
            X_te_df = self.df.loc[test_mask,  self.feature_cols]
            y_tr = self.df.loc[train_mask, self.target_col].values.astype(int)
            y_te = self.df.loc[test_mask,  self.target_col].values.astype(int)

            # Scale: fit ONLY on training data — critical, no leakage
            # After scaling we rebuild DataFrames to preserve column names for LightGBM
            if needs_scaling:
                scaler  = StandardScaler()
                X_tr_df = pd.DataFrame(
                    scaler.fit_transform(X_tr_df),
                    columns=self.feature_cols, index=X_tr_df.index
                )
                X_te_df = pd.DataFrame(
                    scaler.transform(X_te_df),
                    columns=self.feature_cols, index=X_te_df.index
                )

            # ── Diagnose class distribution per fold ────────────────────
            unique, counts = np.unique(y_tr, return_counts=True)
            class_dist = dict(zip(unique, counts))
            rare_classes = [c for c, n in class_dist.items() if n < 30]
            if rare_classes:
                names = [REGIME_NAMES[c] for c in rare_classes]
                logger.warning(
                    f"  {year}: Rare classes in training {names} "
                    f"— counts: {class_dist}. "
                    f"Model may predict zero instances of these classes."
                )

            model = model_factory()
            _fit_with_balanced_weights(model, X_tr_df, y_tr, model_name=model_name)

            y_pred = model.predict(X_te_df)
            if getattr(model, "_label_needs_remap", False):
                reverse = model._label_remap_reverse
                y_pred  = np.array([reverse.get(int(p), int(p)) for p in y_pred])
            proba = model.predict_proba(X_te_df) if hasattr(model, "predict_proba") else None
            # Expand proba to full 3-column array if fold was missing a class
            if proba is not None and getattr(model, "_label_needs_remap", False):
                full_proba = np.zeros((len(proba), 3))
                for remapped_idx, orig_class in model._label_remap_reverse.items():
                    if 0 <= orig_class <= 2:
                        full_proba[:, orig_class] = proba[:, remapped_idx]
                proba = full_proba

            # ── Post-predict rare-class check ────────────────────────────
            # If model predicted ZERO of a class that exists in test set,
            # log a warning so you can investigate the fold
            test_unique = set(np.unique(y_te))
            pred_unique = set(np.unique(y_pred))
            missed = test_unique - pred_unique
            if missed:
                missed_names = [REGIME_NAMES[c] for c in missed]
                logger.warning(
                    f"  {year}: Model predicted ZERO days of {missed_names} "
                    f"despite them existing in test set. "
                    f"Check HMM labels and class balance for this period."
                )

            # Metrics
            acc     = accuracy_score(y_te, y_pred)
            f1_mac  = f1_score(y_te, y_pred, average="macro",    zero_division=0)
            f1_wtd  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            f1_per  = [
                _class_f1_if_present(y_te, y_pred, 0),
                _class_f1_if_present(y_te, y_pred, 1),
                _class_f1_if_present(y_te, y_pred, 2),
            ]
            tr_acc  = self._transition_accuracy(y_te, y_pred)

            fold_results.append({
                "fold":           year,
                "train_size":     train_mask.sum(),
                "test_size":      test_mask.sum(),
                "accuracy":       acc,
                "f1_macro":       f1_mac,
                "f1_weighted":    f1_wtd,
                "f1_bear":        f1_per[0],
                "f1_sideways":    f1_per[1],
                "f1_bull":        f1_per[2],
                "support_bear":     int(np.sum(y_te == 0)),
                "support_sideways": int(np.sum(y_te == 1)),
                "support_bull":     int(np.sum(y_te == 2)),
                "transition_acc": tr_acc,
            })

            # Build prediction DataFrame
            pred_df = self.df.loc[test_mask, ["Close", "target_regime_name"]].copy()
            pred_df["y_true"]      = y_te
            pred_df["y_pred"]      = y_pred
            pred_df["pred_regime"] = pd.Series(y_pred, index=pred_df.index).map(REGIME_NAMES)

            if proba is not None:
                # model.classes_ tells us which classes exist in THIS fold.
                # If a fold has only 2 of 3 classes, proba has shape (n,2) not (n,3).
                # Map by model.classes_ position, not by REGIME_NAMES index.
                model_classes = list(model.classes_) if hasattr(model, "classes_") else list(REGIME_NAMES.keys())
                # Initialize all prob columns to 0.0 first
                for name in REGIME_NAMES.values():
                    pred_df[f"prob_{name.lower()}"] = 0.0
                # Fill only the classes present in this fold
                for col_idx, class_int in enumerate(model_classes):
                    if class_int in REGIME_NAMES:
                        name = REGIME_NAMES[class_int]
                        pred_df[f"prob_{name.lower()}"] = proba[:, col_idx]

                # ── TRANSITION ALERT ─────────────────────────────────────────
                # Flag when top-class confidence drops below threshold.
                # This is the model saying "I'm not sure" — which IS the signal.
                top_prob = proba.max(axis=1)
                pred_df["transition_alert"] = (top_prob < self.alert_thresh).astype(int)

                # Alert severity: how far below threshold?
                # np.clip(a, a_min, a_max) — use this, not pandas .clip(lower=) on numpy arrays
                pred_df["alert_severity"] = np.clip(self.alert_thresh - top_prob, 0, None)

                # Consecutive alert days: sustained uncertainty = stronger signal
                alert_series = pred_df["transition_alert"]
                consecutive  = alert_series.groupby(
                    (alert_series != alert_series.shift()).cumsum()
                ).cumcount() + 1
                pred_df["consecutive_alert_days"] = consecutive * alert_series

            all_preds_list.append(pred_df)

            if verbose:
                alert_days = pred_df.get("transition_alert", pd.Series(0)).sum()
                logger.info(
                    f"  {year} | acc={acc:.3f} | f1_macro={f1_mac:.3f} | "
                    f"f1_bear={f1_per[0]:.3f} | bear_days={int(np.sum(y_te == 0))} | "
                    f"trans_acc={tr_acc:.3f} | "
                    f"alert_days={int(alert_days)}"
                )

        return pd.DataFrame(fold_results), pd.concat(all_preds_list)

    @staticmethod
    def _transition_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_transitions = np.where(np.diff(y_true) != 0)[0] + 1
        if len(true_transitions) == 0:
            return np.nan
        correct = sum(
            any(y_pred[max(0, t-1): min(len(y_pred), t+2)] == y_true[t])
            for t in true_transitions
        )
        return correct / len(true_transitions)


# ────────────────────────────────────────────────────────────────────────────
# MLFLOW EXPERIMENT LOGGER
# ────────────────────────────────────────────────────────────────────────────

def _log_model_safely(model, model_name: str) -> None:
    """
    Log model to MLflow with explicit pip requirements.

    WHY: MLflow's automatic dependency inference (infer_pip_requirements)
    spawns a subprocess that hangs on Python 3.13 due to a threading change
    in subprocess.communicate(). Passing pip_requirements explicitly bypasses
    that subprocess entirely — no hang, no error.

    This is the correct fix regardless of Python version — explicit deps
    are more reliable than auto-inferred ones anyway.
    """
    import sklearn, xgboost, lightgbm
    base_reqs = [
        f"scikit-learn=={sklearn.__version__}",
        f"xgboost=={xgboost.__version__}",
        f"lightgbm=={lightgbm.__version__}",
        "numpy",
        "pandas",
    ]

    try:
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(
                model, "model",
                pip_requirements=base_reqs,
            )
        elif isinstance(model, lgb.LGBMClassifier):
            mlflow.lightgbm.log_model(
                model, "model",
                pip_requirements=base_reqs,
            )
        else:
            mlflow.sklearn.log_model(
                model, "model",
                pip_requirements=base_reqs,
            )
    except Exception as e:
        # If MLflow model logging fails for any reason, log the pickle as artifact
        # Training results are still saved — don't let MLflow logging kill the run
        logger.warning(f"MLflow model logging failed ({e}) — saving as pickle artifact instead")
        import pickle, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, "model_fallback")
        os.unlink(tmp_path)



def run_experiment(
    model_name: str,
    feature_matrix: pd.DataFrame,
    feature_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run a complete WFV experiment for one model, log to MLflow.

    Args:
        model_name     : Key in get_model_registry()
        feature_matrix : Full feature DataFrame
        feature_cols   : Selected feature column names

    Returns:
        fold_results, all_preds, mlflow_run_id
    """
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(registry.keys())}")

    factory, needs_scaling = registry[model_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"Running WFV experiment: {model_name}")
    logger.info(f"{'='*60}")

    validator    = WalkForwardValidator(feature_matrix, feature_cols)
    fold_results, all_preds = validator.run(
        model_factory=factory,
        needs_scaling=needs_scaling,
        model_name=model_name,
    )

    # Fit final model on ALL data for saving (for deployment)
    # Keep as DataFrame so LightGBM stores feature names correctly
    X_all_df = feature_matrix[feature_cols]
    y_all    = feature_matrix["target_regime_code"].values.astype(int)
    scaler   = None
    if needs_scaling:
        scaler   = StandardScaler()
        X_all_df = pd.DataFrame(
            scaler.fit_transform(X_all_df),
            columns=feature_cols,
            index=X_all_df.index,
        )
    final_model = factory()
    final_model.fit(X_all_df, y_all)

    # MLflow logging
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=model_name):
        # Params
        mlflow.log_param("model_type",       model_name)
        mlflow.log_param("n_features",       len(feature_cols))
        mlflow.log_param("wfv_start_year",   cfg.training.wfv_start_year)
        mlflow.log_param("wfv_end_year",     cfg.training.wfv_end_year)
        mlflow.log_param("alert_threshold",  cfg.training.transition_alert_threshold)

        # Aggregate metrics
        agg = fold_results.mean(numeric_only=True)
        for metric in ["accuracy", "f1_macro", "f1_weighted",
                        "f1_bear", "f1_sideways", "f1_bull", "transition_acc"]:
            mlflow.log_metric(f"wfv_{metric}_mean", agg[metric])

        # Per-fold step metrics
        for _, row in fold_results.iterrows():
            step = int(row["fold"])
            mlflow.log_metric("fold_accuracy",       row["accuracy"],       step=step)
            mlflow.log_metric("fold_f1_macro",       row["f1_macro"],       step=step)
            mlflow.log_metric("fold_transition_acc", row["transition_acc"], step=step)

        # Transition alert stats
        if "transition_alert" in all_preds.columns:
            alert_rate = all_preds["transition_alert"].mean()
            mlflow.log_metric("alert_rate", alert_rate)
            logger.info(f"  Transition alert rate: {alert_rate*100:.1f}% of test days")

        # Save fold results as artifact
        results_path = f"data/wfv_{model_name}.csv"
        fold_results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Log model — use explicit pip_requirements to avoid Python 3.13
        # subprocess hang bug in MLflow's automatic dependency inference
        _log_model_safely(final_model, model_name)
        run_id = mlflow.active_run().info.run_id

    logger.info(f"MLflow run: {run_id[:8]}...")

    return fold_results, all_preds, final_model, scaler, run_id


# ────────────────────────────────────────────────────────────────────────────
# MODEL SAVING / LOADING
# ────────────────────────────────────────────────────────────────────────────

def save_best_model(
    model,
    scaler,
    model_name: str,
    feature_cols: list,
    wfv_metrics: dict,
    path: str = None,
) -> None:
    """
    Save the deployment bundle: model + scaler + feature list + metadata.
    This single file is all the API needs to serve predictions.
    """
    path = path or cfg.deployment.model_bundle_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model":         model,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "model_name":    model_name,
        "wfv_metrics":   wfv_metrics,
        "regime_names":  REGIME_NAMES,
        "alert_thresh":  cfg.training.transition_alert_threshold,
        "trained_on":    str(datetime.today().date()),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Best model bundle saved → {path}")


def load_model_bundle(path: str = None) -> dict:
    """Load the deployment bundle for inference."""
    path = path or cfg.deployment.model_bundle_path
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"Model bundle loaded: {bundle['model_name']} "
                f"(trained {bundle['trained_on']})")
    return bundle


def predict_with_alert(
    bundle: dict,
    feature_row: pd.DataFrame,
) -> dict:
    """
    Run inference on a single row of features.
    Returns regime prediction + probabilities + transition alert.

    This is what the API calls on every request.

    Args:
        bundle      : From load_model_bundle()
        feature_row : Single-row DataFrame with correct feature columns

    Returns:
        Dict with: regime, probabilities, transition_alert, alert_severity
    """
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    alert_thresh = bundle["alert_thresh"]
    regime_names = bundle["regime_names"]

    X = feature_row[feature_cols].values
    if scaler is not None:
        X = scaler.transform(X)

    pred_code = int(model.predict(X)[0])
    regime    = regime_names[pred_code]

    proba        = model.predict_proba(X)[0]
    top_prob     = float(proba.max())
    alert_active = top_prob < alert_thresh
    severity     = max(0.0, alert_thresh - top_prob)

    return {
        "regime":            regime,
        "regime_code":       pred_code,
        "prob_bear":         float(proba[0]),
        "prob_sideways":     float(proba[1]),
        "prob_bull":         float(proba[2]),
        "top_confidence":    top_prob,
        "transition_alert":  alert_active,
        "alert_severity":    round(severity, 4),
        "alert_message":     (
            f"⚠️  Regime uncertainty detected — confidence {top_prob*100:.0f}% "
            f"(below {alert_thresh*100:.0f}% threshold). Possible regime transition."
            if alert_active else None
        ),
    }
