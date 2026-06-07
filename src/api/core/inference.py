from typing import Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.schemas import RawMarketPayload, OHLCVPoint, ClosePoint
from src.data.labeler import REGIME_CODE, compute_hmm_inputs
from src.features.pipeline import build_features
from src.models.trainer import predict_with_alert
from src.models.transition_detector import combine_predictions, build_transition_features
from src.utils.logger import get_logger
logger = get_logger(__name__)

HMM_FEATURE_COLS = [
    "trend_5d",
    "trend_21d",
    "trend_63d",        # NEW
    "realized_vol",
    "vol_ratio",
    "drawdown_63d",
    "drawdown_252d",    # NEW
    "up_day_frac_21d",
    "ma200_distance"  # NEW
]


def _to_price_df(points: Iterable[OHLCVPoint]) -> pd.DataFrame:
    rows = []
    for p in points:
        rows.append(
            {
                "Date": pd.to_datetime(p.date),
                "Open": float(p.open) if p.open is not None else float(p.close),
                "High": float(p.high),
                "Low": float(p.low),
                "Close": float(p.close),
                "Volume": float(p.volume) if p.volume is not None else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise HTTPException(status_code=422, detail="Price history is empty.")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    if (df["High"] < df["Low"]).any():
        raise HTTPException(status_code=422, detail="Invalid OHLC: found High < Low.")
    if (df["Close"] <= 0).any():
        raise HTTPException(status_code=422, detail="Close prices must be positive.")
    return df


def _to_close_df(points: Optional[Iterable[ClosePoint]], name: str) -> Optional[pd.DataFrame]:
    if not points:
        return None
    rows = [{"Date": pd.to_datetime(p.date), "Close": float(p.close)} for p in points]
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    if (df["Close"] <= 0).any():
        raise HTTPException(status_code=422, detail=f"{name} close prices must be positive.")
    return df


def _ensure_feature_frame(features_df: pd.DataFrame, expected_cols: Iterable[str]) -> pd.DataFrame:
    expected_cols = list(expected_cols)
    missing = [c for c in expected_cols if c not in features_df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise HTTPException(
            status_code=422,
            detail=f"Engineered feature mismatch. Missing {len(missing)} columns. First missing: {preview}",
        )

    row = features_df[expected_cols].tail(1)
    if row.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough history after feature engineering. Provide longer raw history.",
        )
    values = row.values.astype(float)
    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Engineered features contain NaN/Inf. Provide cleaner or longer input history.",
        )
    return row


def _label_with_saved_hmm(price_df: pd.DataFrame, hmm_bundle: dict) -> pd.DataFrame:
    hmm_inputs = compute_hmm_inputs(price_df)
    if hmm_inputs.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough price history to compute HMM inputs.",
        )

    model = hmm_bundle["model"]
    scaler = hmm_bundle["scaler"]
    state_map = hmm_bundle["state_map"]

    X = hmm_inputs[HMM_FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    raw_states = model.predict(X_scaled)
    latest = hmm_inputs[HMM_FEATURE_COLS].iloc[-1]
    
    logger.info(
        f"Latest bar HMM features: "
        f"trend_5d={latest['trend_5d']*100:+.2f}% | "
        f"trend_21d={latest['trend_21d']*100:+.2f}% | "
        f"vol={latest['realized_vol']*100:.1f}% | "
        f"drawdown_63d={latest['drawdown_63d']*100:.1f}% | "
        f"up_day_frac={latest['up_day_frac_21d']:.2f} | "
        f"drawdown_252d={latest['drawdown_252d']*100:.1f}% | "
        f"ma200_distance={latest['ma200_distance']*100:.1f}%"
    )

    labeled_df = hmm_inputs.copy()
    labeled_df["regime_name"] = pd.Series(raw_states, index=hmm_inputs.index).map(state_map)
    labeled_df["regime_code"] = labeled_df["regime_name"].map(REGIME_CODE)
    return labeled_df


def _engineer_from_raw(raw_payload: RawMarketPayload, hmm_bundle: dict) -> pd.DataFrame:
    price_df = _to_price_df(raw_payload.price)
    vix_df = _to_close_df(raw_payload.vix, "vix")
    usdinr_df = _to_close_df(raw_payload.usdinr, "usdinr")
    crude_df = _to_close_df(raw_payload.crude, "crude")

    if vix_df is None:
        raise HTTPException(status_code=422, detail="VIX history is required.")

    labeled_df = _label_with_saved_hmm(price_df, hmm_bundle)
    feature_matrix, _ = build_features(price_df, vix_df, labeled_df, usdinr_df=usdinr_df, crude_df=crude_df)
    return feature_matrix


def run_regime_prediction(main_bundle: dict, hmm_bundle: dict, raw_payload: RawMarketPayload) -> Dict[str, Any]:
    feature_matrix = _engineer_from_raw(raw_payload, hmm_bundle)
    # ADD THIS — see what regime the HMM assigned to the latest bar
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    latest_regime = feature_matrix["regime_name"].iloc[-1] if "regime_name" in feature_matrix.columns else "unknown"
    latest_code   = feature_matrix["regime_code"].iloc[-1] if "regime_code" in feature_matrix.columns else "unknown"
    logger.info(f"HMM assigned latest bar → regime={latest_regime} (code={latest_code})")
    frame = _ensure_feature_frame(feature_matrix, main_bundle["feature_cols"])
    return predict_with_alert(main_bundle, frame)


def run_transition_prediction(
    detector,
    feat_cols: list,
    hmm_bundle: dict,
    raw_payload: RawMarketPayload,
) -> Dict[str, Any]:
    feature_matrix = _engineer_from_raw(raw_payload, hmm_bundle)
    transition_X, _, _ = build_transition_features(feature_matrix, horizon=detector.horizon)
    frame = _ensure_feature_frame(transition_X, feat_cols)
    return detector.predict(frame, feat_cols)


def run_combined_prediction(
    main_bundle: dict,
    detector,
    transition_feat_cols: list,
    hmm_bundle: dict,
    raw_payload: RawMarketPayload,
) -> Dict[str, Any]:
    main_pred = run_regime_prediction(main_bundle, hmm_bundle, raw_payload)
    transition_pred = run_transition_prediction(detector, transition_feat_cols, hmm_bundle, raw_payload)
    return combine_predictions(main_pred, transition_pred)

