from fastapi import APIRouter, Depends

from src.api.dependencies import get_store
from src.api.schemas import HealthResponse
from src.api.core.model_store import ModelStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(store: ModelStore = Depends(get_store)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        main_model_loaded=store.is_main_loaded,
        transition_model_loaded=store.is_transition_loaded,
        hmm_loaded=store.is_hmm_loaded,
    )
