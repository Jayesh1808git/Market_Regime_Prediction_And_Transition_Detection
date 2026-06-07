from datetime import datetime
from typing import Dict, List, Literal, Any, Optional

from pydantic import BaseModel, Field


PredictMode = Literal["regime", "transition", "combined"]


class OHLCVPoint(BaseModel):
    date: datetime
    open: Optional[float] = None
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class ClosePoint(BaseModel):
    date: datetime
    close: float


class RawMarketPayload(BaseModel):
    price: List[OHLCVPoint] = Field(
        ...,
        min_length=260,
        description="Raw price OHLCV history (ascending by date).",
    )
    vix: List[ClosePoint] = Field(
        ...,
        min_length=260,
        description="Raw VIX close history (ascending by date).",
    )
    usdinr: Optional[List[ClosePoint]] = Field(
        default=None,
        description="Optional USDINR close history.",
    )
    crude: Optional[List[ClosePoint]] = Field(
        default=None,
        description="Optional crude close history.",
    )


class BatchPredictPayload(BaseModel):
    mode: PredictMode = Field(
        default="combined",
        description="Prediction mode for all rows in this batch.",
    )
    items: List[RawMarketPayload] = Field(
        ...,
        min_length=1,
        description="List of raw market payloads.",
    )


class HealthResponse(BaseModel):
    status: str
    main_model_loaded: bool
    transition_model_loaded: bool
    hmm_loaded: bool


class VersionResponse(BaseModel):
    app: str
    version: str
    python_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_on: str
    n_features_main: int
    main_alert_threshold: float
    transition_horizon_days: int
    transition_alert_threshold: float


class PredictionResponse(BaseModel):
    result: Dict[str, Any]


class BatchPredictionResponse(BaseModel):
    mode: PredictMode
    results: List[Dict[str, Any]]


class LivePredictRequest(BaseModel):
    mode: PredictMode = Field(
        default="combined",
        description="Prediction mode.",
    )
    lookback_days: int = Field(
        default=600,
        ge=300,
        le=2000,
        description="Calendar days of history to fetch from yfinance.",
    )
    ticker: str = Field(
        default="^NSEI",
        description="Primary price ticker.",
    )
    vix_ticker: str = Field(
        default="^INDIAVIX",
        description="VIX ticker.",
    )
    usdinr_ticker: Optional[str] = Field(
        default="INR=X",
        description="Optional USDINR ticker.",
    )
    crude_ticker: Optional[str] = Field(
        default="BZ=F",
        description="Optional crude ticker.",
    )
