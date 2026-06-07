from fastapi import Request, HTTPException

from src.api.core.model_store import ModelStore


def get_store(request: Request) -> ModelStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="Model store is not initialized.")
    return store

