"""
src/models/transition_detector.py

A DEDICATED MODEL FOR DETECTING REGIME TRANSITIONS.

WHY A SEPARATE MODEL?
  Your main classifier (RF/XGB/LGB) is optimized for steady-state accuracy.
  It learns "given today's features, what regime are we in?" — and it's very
  good at that (95%+ accuracy). But regime TRANSITIONS are a different task:
  "given today's features, will the regime CHANGE in the next N days?"

  These two tasks have conflicting optimal decision boundaries. A single model
  can't maximize both simultaneously. The solution used in production quant
  systems is a two-model ensemble:

    Model 1 (main classifier): What regime are we in?  → high accuracy
    Model 2 (transition detector): Will it change?     → high transition recall

  The final prediction combines both:
    - If transition detector fires → ALERT (regardless of main model confidence)
    - Final regime = main model output
    - Confidence = weighted combination

ARCHITECTURE:
  Binary classification: 1 = regime will change within next N days, 0 = won't
  Uses gradient boosting with heavy weight on the minority class (transitions
  are rare — ~5-15% of days).
  Features are specifically engineered to capture CHANGE signals, not state.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    classification_report
)
import xgboost as xgb

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# TRANSITION-SPECIFIC FEATURES
# These are different from your main feature set.
# They capture RATE OF CHANGE and DIVERGENCE signals.
# ────────────────────────────────────────────────────────────────────────────

def build_transition_features(
    feature_matrix: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Build features and binary target specifically for transition detection.

    Args:
        feature_matrix : Full feature matrix from pipeline.py
        horizon        : Predict if transition happens within next N days

    Returns:
        X        : Transition-specific feature DataFrame
        y        : Binary target (1 = transition within horizon days)
        feat_cols: List of feature column names
    """
    df  = feature_matrix.copy()
    out = pd.DataFrame(index=df.index)

    regime = df["regime_code"]

    # ── TARGET: will regime change within next `horizon` days? ───────────
    # We look ahead horizon days and check if ANY day has a different regime
    future_change = pd.Series(False, index=df.index)
    for h in range(1, horizon + 1):
        future_change = future_change | (regime != regime.shift(-h))
    y = future_change.astype(int)

    # ── VELOCITY FEATURES: rate of change in key indicators ─────────────
    # The derivative of a signal is often more predictive of change than the level

    # Volatility acceleration (vol expanding or contracting?)
    if "realized_vol_21d" in df.columns:
        rv = df["realized_vol_21d"]
        out["vol_velocity_5d"]   = rv.diff(5).shift(1)        # 5-day change in vol
        out["vol_velocity_10d"]  = rv.diff(10).shift(1)       # 10-day change in vol
        out["vol_acceleration"]  = rv.diff(5).diff(5).shift(1) # Change in change (2nd derivative)
        out["vol_expanding"]     = (rv.diff(5) > 0).astype(int).shift(1)

    # VIX velocity
    if "vix_level" in df.columns:
        vix = df["vix_level"]
        out["vix_velocity_3d"]  = vix.diff(3).shift(1)
        out["vix_velocity_10d"] = vix.diff(10).shift(1)
        out["vix_spike"]        = (vix.diff(3) > vix.rolling(63).std()).astype(int).shift(1)

    # ── DIVERGENCE FEATURES: signals pointing in different directions ────
    # When trend and momentum diverge, a change is coming

    # Price vs momentum divergence
    if "ret_5d" in df.columns and "ret_21d" in df.columns:
        out["momentum_divergence"] = (
            np.sign(df["ret_5d"]) != np.sign(df["ret_21d"])
        ).astype(int).shift(1)

    # Short vol vs long vol divergence (vol term structure)
    if "realized_vol_5d" in df.columns and "realized_vol_21d" in df.columns:
        out["vol_term_divergence"] = (
            df["realized_vol_5d"] - df["realized_vol_21d"]
        ).shift(1)
        out["vol_term_expanding"] = (
            out["vol_term_divergence"] > 0
        ).astype(int)

    # MACD histogram direction change (turning point signal)
    if "macd_hist" in df.columns:
        mh = df["macd_hist"]
        out["macd_hist_turning"] = (
            np.sign(mh) != np.sign(mh.shift(1))
        ).astype(int).shift(1)
        out["macd_hist_velocity"] = mh.diff(3).shift(1)

    # ── REGIME STABILITY FEATURES ────────────────────────────────────────

    # Days in current regime (short duration = unstable)
    regime_changes = (regime != regime.shift(1)).astype(int)
    cumsum         = regime_changes.cumsum()
    days_in_regime = cumsum.groupby(cumsum).cumcount()
    out["days_in_regime"]          = days_in_regime.shift(1)
    out["regime_young"]            = (days_in_regime < 10).astype(int).shift(1)
    out["regime_very_young"]       = (days_in_regime < 5).astype(int).shift(1)

    # Recent regime instability
    for w in [5, 10, 20]:
        out[f"n_changes_last_{w}d"] = regime_changes.rolling(w).sum().shift(1)

    # ── CROSS-ASSET STRESS SIGNALS ───────────────────────────────────────

    # Drawdown acceleration (drawdown getting worse faster)
    if "drawdown_252d" in df.columns:
        dd = df["drawdown_252d"]
        out["drawdown_velocity"]    = dd.diff(5).shift(1)
        out["drawdown_accelerating"] = (dd.diff(5) < dd.shift(5).diff(5)).astype(int).shift(1)

    # RSI extreme zones (overbought/oversold → mean reversion likely)
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        out["rsi_overbought"]  = (rsi > 70).astype(int).shift(1)
        out["rsi_oversold"]    = (rsi < 30).astype(int).shift(1)
        out["rsi_velocity_5d"] = rsi.diff(5).shift(1)

    # India-specific: rupee stress (sudden depreciation precedes NSEI selloffs)
    if "usdinr_chg_5d" in df.columns:
        inr_chg = df["usdinr_chg_5d"]
        out["rupee_stress"] = (inr_chg > inr_chg.rolling(63).quantile(0.85)).astype(int).shift(1)

    # Crude shock
    if "crude_chg_5d" in df.columns:
        crude_chg = df["crude_chg_5d"]
        out["crude_shock"] = (crude_chg.abs() > crude_chg.abs().rolling(63).quantile(0.85)).astype(int).shift(1)

    # ── INCLUDE KEY MAIN FEATURES TOO ────────────────────────────────────
    # Some features from the main model are also useful here
    passthrough = [
        "bb_width", "atr_pct", "vix_zscore_252d",
        "golden_cross", "adx", "stoch_k",
        "regime_lag1", "regime_lag2",
    ]
    for col in passthrough:
        if col in df.columns:
            out[col] = df[col]

    # ── CLEANUP ──────────────────────────────────────────────────────────
    out   = out.dropna()
    y     = y.loc[out.index]

    # Remove last `horizon` rows (no valid target)
    out   = out.iloc[:-horizon]
    y     = y.iloc[:-horizon]

    feat_cols = list(out.columns)

    logger.info(
        f"Transition features built: {out.shape} | "
        f"Transition rate: {y.mean()*100:.1f}% of days | "
        f"Horizon: {horizon} days"
    )
    return out, y, feat_cols


# ────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD TRANSITION DETECTOR
# ────────────────────────────────────────────────────────────────────────────

class TransitionDetector:
    """
    Walk-forward trained binary classifier for regime transition detection.

    Optimizes for RECALL over precision — it's better to fire a false alert
    than to miss a real regime change. This is the opposite of the main model
    which optimizes for accuracy.

    The threshold is tunable: lower threshold = more alerts, higher recall,
    lower precision. Default 0.35 is deliberately sensitive.
    """

    def __init__(
        self,
        alert_threshold: float = 0.35,   # Lower than main model — we want sensitivity
        horizon: int = 5,                 # Predict transitions within next N days
    ):
        self.alert_threshold = alert_threshold
        self.horizon         = horizon
        self.fold_results    = []
        self.final_model     = None
        self.final_scaler    = None

    def _make_model(self):
        """XGBoost with heavy minority-class weighting."""
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,           # Shallow — prevents memorizing specific regimes
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=6,    # ~6:1 non-transition:transition ratio — upweights transitions
            eval_metric="aucpr",   # Area under Precision-Recall curve — right metric for imbalanced
            random_state=cfg.training.random_seed,
            n_jobs=-1,
        )

    def run_wfv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feat_cols: list,
    ) -> pd.DataFrame:
        """
        Walk-forward validation for the transition detector.
        Returns fold results + out-of-sample predictions with probabilities.
        """
        logger.info("\nTraining Transition Detector (Walk-Forward)...")
        all_preds_list = []

        for test_year in range(cfg.training.wfv_start_year, cfg.training.wfv_end_year + 1):
            train_mask = X.index.year < test_year
            test_mask  = X.index.year == test_year

            if train_mask.sum() < 252 * 3 or test_mask.sum() < 30:
                continue

            X_tr = X.loc[train_mask, feat_cols]
            y_tr = y.loc[train_mask].values
            X_te = X.loc[test_mask,  feat_cols]
            y_te = y.loc[test_mask].values

            scaler = StandardScaler()
            X_tr   = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_cols, index=X_tr.index)
            X_te   = pd.DataFrame(scaler.transform(X_te),     columns=feat_cols, index=X_te.index)

            model = self._make_model()
            model.fit(X_tr, y_tr)

            proba  = model.predict_proba(X_te)[:, 1]
            y_pred = (proba >= self.alert_threshold).astype(int)

            # Metrics — recall is the primary metric here
            precision = precision_score(y_te, y_pred, zero_division=0)
            recall    = recall_score(y_te, y_pred, zero_division=0)
            f1        = f1_score(y_te, y_pred, zero_division=0)
            try:
                auc_pr = average_precision_score(y_te, proba)
            except Exception:
                auc_pr = float("nan")

            self.fold_results.append({
                "fold": test_year, "precision": precision,
                "recall": recall, "f1": f1, "auc_pr": auc_pr,
                "n_transitions": y_te.sum(), "n_alerts": y_pred.sum(),
            })

            pred_df = pd.DataFrame({
                "transition_true":  y_te,
                "transition_prob":  proba,
                "transition_alert": y_pred,
            }, index=X.loc[test_mask].index)
            all_preds_list.append(pred_df)

            logger.info(
                f"  {test_year} | precision={precision:.3f} | recall={recall:.3f} | "
                f"f1={f1:.3f} | auc_pr={auc_pr:.3f} | "
                f"true_transitions={y_te.sum()} | alerts_fired={y_pred.sum()}"
            )

        results_df = pd.DataFrame(self.fold_results)
        all_preds  = pd.concat(all_preds_list) if all_preds_list else pd.DataFrame()

        logger.info(f"\nTransition Detector Summary (mean across folds):")
        logger.info(f"  Precision : {results_df['precision'].mean():.3f}")
        logger.info(f"  Recall    : {results_df['recall'].mean():.3f}  ← primary metric")
        logger.info(f"  F1        : {results_df['f1'].mean():.3f}")
        logger.info(f"  AUC-PR    : {results_df['auc_pr'].mean():.3f}")

        return results_df, all_preds

    def fit_final(self, X: pd.DataFrame, y: pd.Series, feat_cols: list) -> None:
        """Fit on all data for deployment."""
        self.final_scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.final_scaler.fit_transform(X[feat_cols]),
            columns=feat_cols, index=X.index
        )
        self.final_model = self._make_model()
        self.final_model.fit(X_scaled, y)
        logger.info("Transition detector fitted on full dataset.")

    def predict(self, X_row: pd.DataFrame, feat_cols: list) -> dict:
        """
        Predict transition probability for a single row.
        Used in API for real-time inference.
        """
        X_scaled = pd.DataFrame(
            self.final_scaler.transform(X_row[feat_cols]),
            columns=feat_cols, index=X_row.index
        )
        prob  = float(self.final_model.predict_proba(X_scaled)[0, 1])
        alert = prob >= self.alert_threshold

        return {
            "transition_probability": round(prob, 4),
            "transition_alert":       alert,
            "alert_strength":         "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.4 else "LOW",
            "message": (
                f"⚠️ Regime transition likely within {self.horizon} days "
                f"(probability: {prob*100:.0f}%)"
                if alert else None
            ),
        }


# ────────────────────────────────────────────────────────────────────────────
# COMBINED PREDICTION — main model + transition detector
# ────────────────────────────────────────────────────────────────────────────

def combine_predictions(
    main_pred: dict,
    transition_pred: dict,
) -> dict:
    """
    Merge main model output with transition detector output.

    Rules:
    - Regime label always comes from the main model (it's more accurate)
    - Alert fires if EITHER model flags uncertainty
    - Confidence is penalized when transition detector fires
    - Alert message prioritizes transition detector when active

    Args:
        main_pred       : Output of predict_with_alert() from trainer.py
        transition_pred : Output of TransitionDetector.predict()

    Returns:
        Combined prediction dict ready for API response / dashboard
    """
    combined_alert   = main_pred["transition_alert"] or transition_pred["transition_alert"]
    transition_prob  = transition_pred["transition_probability"]
    main_confidence  = main_pred["top_confidence"]

    # Penalize confidence when transition detector fires
    adjusted_confidence = main_confidence * (1 - 0.5 * transition_prob)

    # Determine overall alert level
    if transition_prob > 0.6 or (main_confidence < 0.55 and transition_prob > 0.3):
        alert_level = "HIGH"
    elif combined_alert:
        alert_level = "MEDIUM"
    else:
        alert_level = "NONE"

    return {
        # Regime prediction (from main model)
        "regime":              main_pred["regime"],
        "regime_code":         main_pred["regime_code"],
        "prob_bear":           main_pred["prob_bear"],
        "prob_sideways":       main_pred["prob_sideways"],
        "prob_bull":           main_pred["prob_bull"],

        # Confidence (adjusted)
        "raw_confidence":      round(main_confidence, 4),
        "adjusted_confidence": round(adjusted_confidence, 4),

        # Transition signal
        "transition_probability": transition_prob,
        "transition_alert":       combined_alert,
        "alert_level":            alert_level,  # NONE / MEDIUM / HIGH

        # Message for UI
        "alert_message": _build_alert_message(
            alert_level, main_pred, transition_pred
        ),
    }


def _build_alert_message(level: str, main: dict, trans: dict) -> str | None:
    if level == "NONE":
        return None
    regime  = main["regime"]
    conf    = main["top_confidence"]
    t_prob  = trans["transition_probability"]
    horizon = 5  # days

    if level == "HIGH":
        return (
            f"🔴 HIGH ALERT: {regime} regime confidence {conf*100:.0f}% — "
            f"transition probability {t_prob*100:.0f}% within {horizon} days. "
            f"Consider reducing position size."
        )
    return (
        f"🟡 MEDIUM ALERT: {regime} regime confidence {conf*100:.0f}% — "
        f"transition probability {t_prob*100:.0f}%. Monitor closely."
    )


# ────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ────────────────────────name────────────────────────────────────────────────

def save_transition_detector(detector: TransitionDetector, feat_cols: list, path: str = "models/transition_detector.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":           detector.final_model,
        "scaler":          detector.final_scaler,
        "feat_cols":       feat_cols,
        "alert_threshold": detector.alert_threshold,
        "horizon":         detector.horizon,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Transition detector saved → {path}")


def load_transition_detector(path: str = "models/transition_detector.pkl") -> tuple:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    detector = TransitionDetector(
        alert_threshold=bundle["alert_threshold"],
        horizon=bundle["horizon"],
    )
    detector.final_model  = bundle["model"]
    detector.final_scaler = bundle["scaler"]
    logger.info(f"Transition detector loaded from {path}")
    return detector, bundle["feat_cols"]