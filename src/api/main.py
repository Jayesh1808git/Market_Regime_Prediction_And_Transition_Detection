import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Add this import

# Allow direct execution: `python src/api/main.py`
# by ensuring project root (contains `src/`) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.core.model_store import ModelStore
from src.api.routers.health import router as health_router
from src.api.routers.model import router as model_router
from src.api.routers.predict import router as predict_router
from src.api.routers.metrics import router as metrics_router
from src.api.routers.version import router as version_router
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ModelStore()
    try:
        store.load()
        logger.info("API models loaded successfully.")
    except Exception as exc:
        logger.warning(f"API startup model loading failed: {exc}")
    app.state.store = store
    yield


app = FastAPI(
    title="Market Regime Prediction API",
    description="Regime classification + transition detection service.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(version_router)
app.include_router(model_router)
app.include_router(metrics_router)
app.include_router(predict_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
