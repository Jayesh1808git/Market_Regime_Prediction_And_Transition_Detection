import platform

from fastapi import APIRouter

from src.api.schemas import VersionResponse

router = APIRouter(tags=["meta"])


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        app="market-regime-api",
        version="1.0.0",
        python_version=platform.python_version(),
    )

