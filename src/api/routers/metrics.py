from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.core.model_store import ModelStore
from src.api.dependencies import get_store
from src.utils.config import cfg

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/latest")
def latest_metrics(store: ModelStore = Depends(get_store)):
    df = store.latest_metrics_table()
    table = df.to_dict(orient="records") if not df.empty else []

    path = Path(cfg.training.model_comparison_path)
    updated_at = None
    if path.exists():
        updated_at = path.stat().st_mtime

    return {
        "source": str(path),
        "updated_epoch": updated_at,
        "rows": table,
    }

