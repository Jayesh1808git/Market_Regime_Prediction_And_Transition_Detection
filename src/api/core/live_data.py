from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import HTTPException

from src.api.schemas import OHLCVPoint, ClosePoint, RawMarketPayload


def _download_df(ticker: str, lookback_days: int) -> pd.DataFrame:
    period = f"{lookback_days}d"
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=502, detail=f"No market data returned for {ticker}.")
    return df


def _to_ohlcv_points(df: pd.DataFrame) -> list[OHLCVPoint]:
    out = []
    for dt, row in df.iterrows():
        out.append(
            OHLCVPoint(
                date=dt.to_pydatetime(),
                open=float(row.get("Open", row["Close"])),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
            )
        )
    return out


def _to_close_points(df: pd.DataFrame) -> list[ClosePoint]:
    out = []
    for dt, row in df.iterrows():
        out.append(ClosePoint(date=dt.to_pydatetime(), close=float(row["Close"])))
    return out


def fetch_live_payload(
    ticker: str,
    vix_ticker: str,
    lookback_days: int,
    usdinr_ticker: Optional[str] = None,
    crude_ticker: Optional[str] = None,
) -> RawMarketPayload:
    price_df = _download_df(ticker, lookback_days)
    vix_df = _download_df(vix_ticker, lookback_days)

    usdinr_df = _download_df(usdinr_ticker, lookback_days) if usdinr_ticker else None
    crude_df = _download_df(crude_ticker, lookback_days) if crude_ticker else None

    if len(price_df) < 260 or len(vix_df) < 260:
        raise HTTPException(
            status_code=422,
            detail=(
                "Insufficient history fetched for feature engineering. "
                "Increase lookback_days or use tickers with longer history."
            ),
        )

    return RawMarketPayload(
        price=_to_ohlcv_points(price_df),
        vix=_to_close_points(vix_df),
        usdinr=_to_close_points(usdinr_df) if usdinr_df is not None else None,
        crude=_to_close_points(crude_df) if crude_df is not None else None,
    )

