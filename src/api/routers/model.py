from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_store
from src.api.schemas import ModelInfoResponse
from src.api.core.model_store import ModelStore

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info(store: ModelStore = Depends(get_store)) -> ModelInfoResponse:
    if not store.is_main_loaded or not store.is_transition_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    main_bundle = store.main_bundle
    detector = store.transition_detector
    return ModelInfoResponse(
        model_name=main_bundle.get("model_name", "unknown"),
        trained_on=main_bundle.get("trained_on", "unknown"),
        n_features_main=len(main_bundle.get("feature_cols", [])),
        main_alert_threshold=float(main_bundle.get("alert_thresh", 0.0)),
        transition_horizon_days=int(detector.horizon),
        transition_alert_threshold=float(detector.alert_threshold),
    )

