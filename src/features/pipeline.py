"""
src/features/pipeline.py

Responsibility: Take raw OHLCV + VIX + labeled regime data,
return a clean feature matrix ready for model training.

LESSON FOR ANY PROJECT:
  Your feature pipeline should be ONE function call:
      feature_matrix = build_features(spy, vix, labeled_df)
  The caller never needs to know how features are built internally.
  This makes it trivial to add/remove features without touching train.py.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# PRIVATE FEATURE BUILDERS
# Each returns a DataFrame indexed by date.
# All features are shifted by 1 to prevent lookahead.
# ────────────────────────────────────────────────────────────────────────────

def _momentum(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))

    for p in [1, 2, 3, 5, 10, 21, 63]:
        out[f"ret_{p}d"] = np.log(close / close.shift(p)).shift(1)
    for w in [5, 10, 21]:
        out[f"ret_mean_{w}d"] = log_ret.rolling(w).mean().shift(1)
    for p in [5, 21]:
        out[f"roc_{p}d"]     = ((close - close.shift(p)) / close.shift(p)).shift(1)
        out[f"up_frac_{p}d"] = (log_ret > 0).rolling(p).mean().shift(1)

    return out


def _trend(df: pd.DataFrame) -> pd.DataFrame:
    out   = pd.DataFrame(index=df.index)
    close = df["Close"]

    for w in [20, 50, 100, 200]:
        out[f"price_sma{w}_ratio"] = (close / close.rolling(w).mean()).shift(1)

    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50_200_ratio"] = (sma50 / sma200).shift(1)
    out["golden_cross"]    = (sma50 > sma200).astype(int).shift(1)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    out["macd_line"]      = macd["MACD_12_26_9"].shift(1)
    out["macd_signal"]    = macd["MACDs_12_26_9"].shift(1)
    out["macd_hist"]      = macd["MACDh_12_26_9"].shift(1)
    out["macd_crossover"] = (macd["MACD_12_26_9"] > macd["MACDs_12_26_9"]).astype(int).shift(1)

    adx = ta.adx(df["High"], df["Low"], close, length=14)
    out["adx"]     = adx["ADX_14"].shift(1)
    out["adx_pos"] = adx["DMP_14"].shift(1)
    out["adx_neg"] = adx["DMN_14"].shift(1)

    return out


def _volatility(df: pd.DataFrame) -> pd.DataFrame:
    out     = pd.DataFrame(index=df.index)
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))

    for w in [5, 10, 21, 63]:
        out[f"realized_vol_{w}d"] = (log_ret.rolling(w).std() * np.sqrt(252)).shift(1)

    out["vol_ratio_5_21"] = (log_ret.rolling(5).std() / log_ret.rolling(21).std()).shift(1)

    hl_log = np.log(df["High"] / df["Low"])
    out["parkinson_vol"] = (
        (hl_log**2 / (4 * np.log(2))).rolling(21).mean().apply(np.sqrt) * np.sqrt(252)
    ).shift(1)

    atr = ta.atr(df["High"], df["Low"], close, length=14)
    out["atr"]     = atr.shift(1)
    out["atr_pct"] = (atr / close).shift(1)

    bb = ta.bbands(close, length=20, std=2)
    out["bb_width"] = ((bb["BBU_20_2.0_2.0"] - bb["BBL_20_2.0_2.0"]) / bb["BBM_20_2.0_2.0"]).shift(1)
    out["bb_pct_b"] = bb["BBP_20_2.0_2.0"].shift(1)

    return out


def _mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    out   = pd.DataFrame(index=df.index)
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    for p in [7, 14, 28]:
        out[f"rsi_{p}"] = ta.rsi(close, length=p).shift(1)

    for w in [21, 63]:
        roll_mean = close.rolling(w).mean()
        roll_std  = close.rolling(w).std()
        out[f"zscore_{w}d"] = ((close - roll_mean) / roll_std).shift(1)

    for w, col in [(252, "drawdown_252d"), (63, "drawdown_63d")]:
        rolling_high   = close.rolling(w, min_periods=1).max()
        out[col]       = ((close - rolling_high) / rolling_high).shift(1)

    stoch = ta.stoch(df["High"], df["Low"], close, k=14, d=3)
    out["stoch_k"] = stoch["STOCHk_14_3_3"].shift(1)
    out["stoch_d"] = stoch["STOCHd_14_3_3"].shift(1)

    return out


def _macro(df_index: pd.Index, vix_df: pd.DataFrame) -> pd.DataFrame:
    out   = pd.DataFrame(index=df_index)
    vix   = vix_df["Close"].reindex(df_index).ffill()

    out["vix_level"]      = vix.shift(1)
    out["vix_log"]        = np.log(vix).shift(1)
    out["vix_chg_1d"]     = vix.pct_change(1).shift(1)
    out["vix_chg_5d"]     = vix.pct_change(5).shift(1)
    out["vix_chg_21d"]    = vix.pct_change(21).shift(1)

    vix_mean = vix.rolling(252).mean()
    vix_std  = vix.rolling(252).std()
    out["vix_zscore_252d"]  = ((vix - vix_mean) / vix_std).shift(1)
    out["vix_regime_low"]   = (vix < 15).astype(int).shift(1)
    out["vix_regime_high"]  = (vix > 25).astype(int).shift(1)
    out["vix_above_ma21"]   = (vix > vix.rolling(21).mean()).astype(int).shift(1)

    return out


def _regime_lags(labeled_df: pd.DataFrame) -> pd.DataFrame:
    out  = pd.DataFrame(index=labeled_df.index)
    code = labeled_df["regime_code"]

    for lag in [1, 2, 3, 5]:
        out[f"regime_lag{lag}"] = code.shift(lag)

    for lag in [1, 2]:
        lagged = code.shift(lag)
        out[f"was_bull_lag{lag}"]     = (lagged == 2).astype(int)
        out[f"was_sideways_lag{lag}"] = (lagged == 1).astype(int)
        out[f"was_bear_lag{lag}"]     = (lagged == 0).astype(int)

    return out


def _transition_features(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specifically designed to capture imminent regime CHANGES.
    These directly address the low transition accuracy from Notebook 03.

    Key insight: A regime change is signaled by:
    - Rising uncertainty (probability entropy increasing)
    - Volatility expansion (vol_ratio_5_21 > 1)
    - MACD crossover events
    - Drawdown acceleration
    - Recent regime inconsistency (mixed signals in last 5 days)
    """
    out  = pd.DataFrame(index=labeled_df.index)
    code = labeled_df["regime_code"]

    # Days since last regime change — shorter = more volatile regime history
    regime_changes  = (code != code.shift(1)).astype(int)
    cumsum          = regime_changes.cumsum()
    # Count days elapsed in current regime (reset at each change)
    out["days_in_current_regime"] = cumsum.groupby(cumsum).cumcount().shift(1)

    # Binary: did regime change in the last N days?
    for w in [3, 5, 10]:
        out[f"regime_changed_last_{w}d"] = regime_changes.rolling(w).max().shift(1)

    # Regime instability: number of different regimes in last 10 days
    out["regime_nunique_10d"] = (
        pd.Series(code).rolling(10).apply(lambda x: len(set(x)), raw=True).shift(1)
    )

    # Transition label (used in transition-specific secondary model)
    # 1 if regime changes within next 5 days, else 0
    future_code = code.shift(-1)
    out["transition_imminent"] = (code != future_code).astype(int).shift(1)

    return out


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API — this is the only function callers need
# ────────────────────────────────────────────────────────────────────────────

def build_features(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    usdinr_df: pd.DataFrame = None,
    crude_df: pd.DataFrame = None,
) -> tuple[pd.DataFrame, list]:
    """
    Build the complete feature matrix from raw inputs.

    Args:
        spy_df     : Raw OHLCV for NSEI (or SPY)
        vix_df     : Raw OHLCV for India VIX (or ^VIX)
        labeled_df : HMM-labeled DataFrame (from labeler.py)
        usdinr_df  : USD/INR OHLCV — optional, India-specific
        crude_df   : Brent crude OHLCV — optional, India-specific

    Returns:
        feature_matrix : Full DataFrame with features + target columns
        feature_cols   : List of feature column names (excludes targets)
    """
    logger.info("Building feature matrix...")

    blocks = [
        labeled_df[["Close", "regime_code", "regime_name"]],
        _momentum(spy_df),
        _trend(spy_df),
        _volatility(spy_df),
        _mean_reversion(spy_df),
        _macro(spy_df.index, vix_df),
        _regime_lags(labeled_df),
        _transition_features(labeled_df),
    ]

    # Add India-specific macro block if data is provided
    if usdinr_df is not None or crude_df is not None:
        india_block = _india_macro(spy_df.index, usdinr_df, crude_df)
        blocks.append(india_block)
        logger.info("  Added India macro features (USD/INR, Brent crude)")

    df = pd.concat(blocks, axis=1)
    df = df.loc[labeled_df.index]

    # ── Create target columns (1-day ahead) ──────────────────────────────
    df["target_regime_code"] = df["regime_code"].shift(-1)
    df["target_regime_name"] = df["regime_name"].shift(-1)

    # ── Drop warmup rows and NaN targets ────────────────────────────────
    df = df.dropna(subset=["target_regime_code"])
    feature_cols = [
        c for c in df.columns
        if c not in ["Close", "regime_code", "regime_name",
                     "target_regime_code", "target_regime_name",
                     "transition_imminent"]
    ]

    nan_frac = df[feature_cols].isnull().mean(axis=1)
    df = df[nan_frac < 0.20]
    df[feature_cols] = df[feature_cols].ffill().bfill()

    logger.info(f"Feature matrix built: {df.shape} | {len(feature_cols)} features")
    return df, feature_cols


def select_features(
    feature_matrix: pd.DataFrame,
    feature_cols: list,
    importances: pd.Series,
    threshold: float = None,
) -> list:
    """
    Remove highly correlated features, keeping higher-importance one.

    Args:
        feature_matrix : Full feature matrix
        feature_cols   : All candidate feature names
        importances    : Feature importance scores (from quick RF)
        threshold      : Correlation threshold (default: from config)

    Returns:
        List of selected (pruned) feature names
    """
    threshold = threshold or cfg.features.corr_threshold
    corr      = feature_matrix[feature_cols].corr().abs()
    to_drop   = set()

    for i, col_i in enumerate(feature_cols):
        if col_i in to_drop:
            continue
        for col_j in feature_cols[i+1:]:
            if col_j in to_drop:
                continue
            if corr.loc[col_i, col_j] > threshold:
                drop = col_j if importances.get(col_i, 0) >= importances.get(col_j, 0) else col_i
                to_drop.add(drop)

    selected = [c for c in feature_cols if c not in to_drop]
    logger.info(f"Feature selection: {len(feature_cols)} → {len(selected)} "
                f"(removed {len(to_drop)} correlated)")
    return selected


def save_features(
    feature_matrix: pd.DataFrame,
    selected_features: list,
    matrix_path: str = None,
    features_path: str = None,
) -> None:
    """Save feature matrix and selected feature list to disk."""
    matrix_path   = matrix_path   or cfg.features.feature_matrix_path
    features_path = features_path or cfg.features.selected_features_path

    Path(matrix_path).parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(matrix_path)
    with open(features_path, "wb") as f:
        pickle.dump(selected_features, f)

    logger.info(f"Feature matrix saved → {matrix_path}")
    logger.info(f"Selected features saved → {features_path}")


def load_features(
    matrix_path: str = None,
    features_path: str = None,
) -> tuple[pd.DataFrame, list]:
    """Load saved feature matrix and selected feature list."""
    matrix_path   = matrix_path   or cfg.features.feature_matrix_path
    features_path = features_path or cfg.features.selected_features_path

    df = pd.read_csv(matrix_path, index_col=0, parse_dates=True)
    with open(features_path, "rb") as f:
        selected = pickle.load(f)

    logger.info(f"Features loaded: {df.shape} | {len(selected)} selected features")
    return df, selected


def _india_macro(df_index: pd.Index, usdinr_df: pd.DataFrame, crude_df: pd.DataFrame) -> pd.DataFrame:
    """
    India-specific macro features.

    WHY THESE MATTER FOR NSEI:
    - USD/INR: When rupee weakens (INR=X rises), Foreign Institutional Investors
      (FIIs) sell Indian equities to avoid currency losses. FII selling = Bear regime.
      This is the single most India-specific signal — it has no SPY equivalent.

    - Brent crude: India imports ~85% of its oil. A crude spike directly raises
      inflation, compresses corporate margins, and forces RBI to hike rates.
      Sustained crude rise historically precedes NSEI Bear regimes.
    """
    out = pd.DataFrame(index=df_index)

    # ── USD/INR Features ────────────────────────────────────────────────
    if usdinr_df is not None and not usdinr_df.empty:
        inr = usdinr_df["Close"].reindex(df_index).ffill()

        out["usdinr_level"]      = inr.shift(1)
        out["usdinr_chg_1d"]     = inr.pct_change(1).shift(1)
        out["usdinr_chg_5d"]     = inr.pct_change(5).shift(1)
        out["usdinr_chg_21d"]    = inr.pct_change(21).shift(1)

        # Rupee strength z-score — how weak/strong vs trailing 1 year
        inr_mean = inr.rolling(252).mean()
        inr_std  = inr.rolling(252).std()
        out["usdinr_zscore"]     = ((inr - inr_mean) / inr_std).shift(1)

        # Rupee above 21-day MA = weakening trend = bearish for NSEI
        out["rupee_weakening"]   = (inr > inr.rolling(21).mean()).astype(int).shift(1)

        # Rupee momentum: 5d return > 0 means rupee is depreciating (bad for FIIs)
        out["rupee_depreciation_5d"] = (inr.pct_change(5) > 0).astype(int).shift(1)

    # ── Brent Crude Features ─────────────────────────────────────────────
    if crude_df is not None and not crude_df.empty:
        crude = crude_df["Close"].reindex(df_index).ffill()

        out["crude_level"]       = crude.shift(1)
        out["crude_chg_5d"]      = crude.pct_change(5).shift(1)
        out["crude_chg_21d"]     = crude.pct_change(21).shift(1)

        # Crude z-score vs 1-year rolling window
        crude_mean = crude.rolling(252).mean()
        crude_std  = crude.rolling(252).std()
        out["crude_zscore"]      = ((crude - crude_mean) / crude_std).shift(1)

        # High crude regime (> 1-yr rolling 75th percentile) — inflationary pressure
        crude_75pct = crude.rolling(252).quantile(0.75)
        out["crude_high_regime"] = (crude > crude_75pct).astype(int).shift(1)

        # Crude spike: > 10% jump in 5 days — historically triggers NSEI selloffs
        out["crude_spike_5d"]    = (crude.pct_change(5) > 0.10).astype(int).shift(1)

    return out