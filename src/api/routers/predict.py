from fastapi import APIRouter, Depends, HTTPException

from src.api.core.inference import (
    run_regime_prediction,
    run_transition_prediction,
    run_combined_prediction,
)
from src.api.core.live_data import fetch_live_payload
from src.api.core.model_store import ModelStore
from src.api.dependencies import get_store
from src.api.schemas import (
    RawMarketPayload,
    BatchPredictPayload,
    LivePredictRequest,
    PredictionResponse,
    BatchPredictionResponse,
)
from src.utils.config import cfg

router = APIRouter(prefix="/predict", tags=["predict"])


def _assert_models(store: ModelStore) -> None:
    if not store.is_main_loaded:
        raise HTTPException(status_code=503, detail="Main model is not loaded.")
    if not store.is_transition_loaded:
        raise HTTPException(status_code=503, detail="Transition model is not loaded.")
    if not store.is_hmm_loaded:
        raise HTTPException(status_code=503, detail="HMM artifacts are not loaded.")


@router.post("/regime", response_model=PredictionResponse)
def predict_regime(payload: RawMarketPayload, store: ModelStore = Depends(get_store)) -> PredictionResponse:
    _assert_models(store)
    result = run_regime_prediction(store.main_bundle, store.hmm_bundle, payload)
    return PredictionResponse(result=result)


@router.post("/transition", response_model=PredictionResponse)
def predict_transition(payload: RawMarketPayload, store: ModelStore = Depends(get_store)) -> PredictionResponse:
    _assert_models(store)
    result = run_transition_prediction(
        store.transition_detector,
        store.transition_feat_cols,
        store.hmm_bundle,
        payload,
    )
    return PredictionResponse(result=result)


@router.post("/combined", response_model=PredictionResponse)
def predict_combined(payload: RawMarketPayload, store: ModelStore = Depends(get_store)) -> PredictionResponse:
    _assert_models(store)
    result = run_combined_prediction(
        store.main_bundle,
        store.transition_detector,
        store.transition_feat_cols,
        store.hmm_bundle,
        payload,
    )
    return PredictionResponse(result=result)


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictPayload, store: ModelStore = Depends(get_store)) -> BatchPredictionResponse:
    _assert_models(store)

    results = []
    for item in payload.items:
        if payload.mode == "regime":
            out = run_regime_prediction(store.main_bundle, store.hmm_bundle, item)
        elif payload.mode == "transition":
            out = run_transition_prediction(
                store.transition_detector,
                store.transition_feat_cols,
                store.hmm_bundle,
                item,
            )
        else:
            out = run_combined_prediction(
                store.main_bundle,
                store.transition_detector,
                store.transition_feat_cols,
                store.hmm_bundle,
                item,
            )
        results.append(out)

    return BatchPredictionResponse(mode=payload.mode, results=results)


@router.post("/live", response_model=PredictionResponse)
def predict_live(payload: LivePredictRequest, store: ModelStore = Depends(get_store)) -> PredictionResponse:
    _assert_models(store)

    raw_payload = fetch_live_payload(
        ticker=payload.ticker or cfg.data.ticker,
        vix_ticker=payload.vix_ticker or cfg.data.vix_ticker,
        lookback_days=payload.lookback_days,
        usdinr_ticker=payload.usdinr_ticker,
        crude_ticker=payload.crude_ticker,
    )

    if payload.mode == "regime":
        out = run_regime_prediction(store.main_bundle, store.hmm_bundle, raw_payload)
    elif payload.mode == "transition":
        out = run_transition_prediction(
            store.transition_detector,
            store.transition_feat_cols,
            store.hmm_bundle,
            raw_payload,
        )
    else:
        out = run_combined_prediction(
            store.main_bundle,
            store.transition_detector,
            store.transition_feat_cols,
            store.hmm_bundle,
            raw_payload,
        )

    # Include metadata and recent series so frontend can render context charts.
    last_dt = raw_payload.price[-1].date.isoformat() if raw_payload.price else None
    price_tail = [
        {"date": p.date.isoformat(), "close": float(p.close)}
        for p in raw_payload.price[-90:]
    ]
    vix_tail = [
        {"date": v.date.isoformat(), "close": float(v.close)}
        for v in raw_payload.vix[-90:]
    ]
    out["live_data_meta"] = {
        "mode": payload.mode,
        "ticker": payload.ticker,
        "vix_ticker": payload.vix_ticker,
        "usdinr_ticker": payload.usdinr_ticker,
        "crude_ticker": payload.crude_ticker,
        "lookback_days": payload.lookback_days,
        "latest_bar_time": last_dt,
        "price_tail": price_tail,
        "vix_tail": vix_tail,
    }
    return PredictionResponse(result=out)
