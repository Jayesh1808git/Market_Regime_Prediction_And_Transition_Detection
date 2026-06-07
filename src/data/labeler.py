"""
src/data/labeler.py

WHY THE ORIGINAL LABELER FAILED ON NSEI:
  The original used only 3 features: log_return, realized_vol, daily_range.
  For NSEI, these are insufficient because:

  1. log_return is noisy at daily frequency — daily returns are ~50% noise.
  2. Volatility alone can't separate Bull from Bear on NSEI — both regimes
     have high volatility. The HMM was essentially doing: low vol = Sideways,
     high vol = randomly Bear OR Bull → produced the wrong 40/40 Bear/Bull split.
  3. No trend direction feature — vol tells you HOW MUCH the market moves,
     not WHERE it goes. Without trend, HMM can't distinguish volatile-bull
     from volatile-bear.

FIX: 6 richer features capturing both DIRECTION and MAGNITUDE at multiple horizons.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)

REGIME_CODE = {"Bear": 0, "Sideways": 1, "Bull": 2}

# Expected distribution for NSEI (2007-2024).
#
# NSEI-SPECIFIC REALITY (different from SPY):
# The HMM naturally finds 3 clusters:
#   Bull:     low drawdown (-1%), low vol (11%), strong trend  → ~40-55% of days
#   Sideways: moderate drawdown (-5%), moderate vol (18%)      → ~35-50% of days
#             This covers corrections, consolidations, mild bears
#   Bear:     severe drawdown (-15%+), high vol (40%+)         → ~8-15% of days
#             Only true crash regimes: 2008 GFC, 2020 COVID
#
# Key insight: With only 3 states, NSEI cannot cleanly separate "Sideways" from
# "mild Bear". The middle state is a mixed regime. This is expected and acceptable.
EXPECTED_DIST = {
    "Bull":     (0.35, 0.75),  # NSEI is structurally bullish — 66% is realistic
    "Sideways": (0.15, 0.40),
    "Bear":     (0.05, 0.20),
}


def compute_hmm_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 6 rich features for HMM regime labeling.

    Feature design: capture BOTH direction AND magnitude at MULTIPLE horizons.
    Each feature targets a specific aspect of market regime that the original
    3-feature set missed.
    """
    out   = pd.DataFrame(index=df.index)
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))

    # Direction features — multiple horizons
    out["trend_5d"]         = log_ret.rolling(5).sum().rolling(3).mean()
    out["trend_21d"]        = log_ret.rolling(21).sum()
    out["trend_63d"]        = log_ret.rolling(63).sum()   # quarterly trend

    # Magnitude features
    out["realized_vol"]     = log_ret.rolling(cfg.hmm.vol_window).std() * np.sqrt(252)
    short_vol               = log_ret.rolling(5).std()
    long_vol                = log_ret.rolling(21).std().replace(0, np.nan)
    out["vol_ratio"]        = (short_vol / long_vol).clip(0.2, 5.0)

    # Structural features — two drawdown horizons
    # 63d captures recent pullbacks; 126d captures extended bear markets
    rolling_high_63         = close.rolling(63,  min_periods=1).max()
    rolling_high_252        = close.rolling(252, min_periods=1).max()
    out["drawdown_63d"]     = (close - rolling_high_63)  / rolling_high_63
    out["drawdown_252d"]    = (close - rolling_high_252) / rolling_high_252
    out["up_day_frac_21d"]  = (log_ret > 0).rolling(21).mean()
    ma200 = close.rolling(200, min_periods=1).mean()
    out["ma200_distance"] = (close - ma200) / ma200

    out["Close"] = close
    out = out.dropna()

    logger.info(f"HMM input features computed: {out.shape[0]} observations")
    return out


def fit_hmm(hmm_inputs: pd.DataFrame) -> tuple:
    """
    Fit Gaussian HMM with multiple seeds (picks best log-likelihood).
    Includes label smoothing and distribution validation.
    """
    feature_cols = [
        "trend_5d", "trend_21d", "trend_63d", "realized_vol",
        "vol_ratio", "drawdown_63d", "drawdown_252d", "up_day_frac_21d","ma200_distance"
    ]
    X        = hmm_inputs[feature_cols].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Multiple seeds — HMM is sensitive to initialization
    best_model, best_score = None, -np.inf
    seeds = [cfg.hmm.random_seed, 0, 7, 13, 99]

    for seed in seeds:
        model = GaussianHMM(
            n_components=cfg.hmm.n_regimes,
            covariance_type="full",
            n_iter=2000,
            tol=1e-5,
            random_state=seed,
            verbose=False,
        )
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score, best_model = score, model
        except Exception as e:
            logger.warning(f"  HMM seed {seed} failed: {e}")

    model      = best_model
    raw_states = model.predict(X_scaled)

    logger.info(f"HMM converged: {model.monitor_.converged} | "
                f"Log-likelihood: {best_score:.2f} (best of {len(seeds)} seeds)")

    # Sort by mean drawdown_63d: most negative = Bear, middle = Sideways, least = Bull
    #
    # WHY DRAWDOWN NOT TREND:
    # trend_21d alone fails when two states have similar (both slightly negative) trends
    # but very different severities — e.g. a mild Bear (-0.8%) vs a flat Sideways (-0.3%).
    # Drawdown cleanly separates all three:
    #   Bull:     drawdown ≈ -1%   (market near recent highs)
    #   Sideways: drawdown ≈ -3%   (mild pullback, consolidating)
    #   Bear:     drawdown ≈ -8%+  (meaningful decline from peak)
    # This is also the most economically meaningful separator — drawdown directly
    # measures how far the market has fallen from its recent high.
    state_drawdowns = {
        s: hmm_inputs.iloc[raw_states == s]["drawdown_252d"].mean()
        for s in range(cfg.hmm.n_regimes)
    }
    sorted_states = sorted(state_drawdowns, key=state_drawdowns.get)  # most negative first
    regime_names  = {0: "Bear", 1: "Sideways", 2: "Bull"}
    state_map     = {raw: regime_names[rank] for rank, raw in enumerate(sorted_states)}

    for raw_s, name in state_map.items():
        mask = raw_states == raw_s
        logger.info(
            f"  State {raw_s} → {name:8s} | "
            f"trend_21d={hmm_inputs.iloc[mask]['trend_21d'].mean()*100:+.2f}% | "
            f"vol={hmm_inputs.iloc[mask]['realized_vol'].mean()*100:.1f}% | "
            f"drawdown={hmm_inputs.iloc[mask]['drawdown_63d'].mean()*100:.1f}% | "
            f"n_days={mask.sum()}"
            f"drawdown_252d={hmm_inputs.iloc[mask]['drawdown_252d'].mean()*100:.1f}%"
            f"ma200_distance={hmm_inputs.iloc[mask]['ma200_distance'].mean()*100:.1f}%"
        )

    # Smooth short isolated spikes (< 3 days) — reduces noise in labels
    raw_labels = pd.Series(raw_states, index=hmm_inputs.index)
    raw_labels = _smooth_labels(raw_labels, min_duration=3)

    labeled_df = hmm_inputs.copy()
    labeled_df["regime_name"] = raw_labels.map(state_map)
    labeled_df["regime_code"] = labeled_df["regime_name"].map(REGIME_CODE)

    _validate_distribution(labeled_df["regime_name"])

    return labeled_df, model, scaler, state_map


def _smooth_labels(labels: pd.Series, min_duration: int = 3) -> pd.Series:
    """
    Remove regime spikes shorter than min_duration days.
    A 1-2 day regime is almost always noise, not a real transition.

    Before: Bull Bull Bear Bull Bull  (1-day Bear spike)
    After:  Bull Bull Bull Bull Bull  (smoothed)
    """
    arr = labels.values.copy()
    i   = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        if (j - i) < min_duration and i > 0:
            arr[i:j] = arr[i - 1]
        i = j
    return pd.Series(arr, index=labels.index)


def _validate_distribution(regime_names: pd.Series) -> None:
    """Warn if regime distribution is outside expected bounds for NSEI."""
    dist   = regime_names.value_counts(normalize=True)
    all_ok = True

    logger.info("Regime distribution validation:")
    for regime, (low, high) in EXPECTED_DIST.items():
        actual = dist.get(regime, 0.0)
        ok     = low <= actual <= high
        logger.info(
            f"  {regime:8s}: {actual*100:.1f}%  "
            f"(expected {low*100:.0f}-{high*100:.0f}%)  "
            f"{'OK' if ok else 'OUT OF RANGE ⚠'}"
        )
        if not ok:
            all_ok = False

    if not all_ok:
        logger.warning(
            "HMM distribution outside expected bounds — likely mislabeling. "
            "If Bear > 35% or Bull < 35%, the HMM is confusing volatile-bull "
            "with bear. Delete data/hmm_model.pkl and re-run."
        )


def save_hmm_artifacts(model, scaler, state_map, path: str = None) -> None:
    path = path or cfg.hmm.model_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "state_map": state_map}, f)
    logger.info(f"HMM artifacts saved → {path}")


def load_hmm_artifacts(path: str = None) -> dict:
    path = path or cfg.hmm.model_path
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"HMM artifacts loaded from {path}")
    return bundle