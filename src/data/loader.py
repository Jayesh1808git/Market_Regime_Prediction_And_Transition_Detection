"""
src/data/loader.py

Responsibility: Fetch and cache raw market data. Nothing else.
This module does NOT compute features, does NOT train models.

LESSON FOR ANY PROJECT:
  Your data layer's only job is to return clean DataFrames.
  Always cache to disk — re-downloading on every run wastes time.
  Always validate what you got before returning it.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import yfinance as yf

from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_path: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data. Uses cached CSV if available.

    Args:
        ticker        : Yahoo Finance ticker symbol (e.g. 'SPY', '^VIX')
        start         : Start date string 'YYYY-MM-DD'
        end           : End date string 'YYYY-MM-DD'
        cache_path    : Path to cache CSV. If None, no caching.
        force_refresh : Re-download even if cache exists.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
        Index is DatetimeIndex of trading days.
    """
    # Return cache if it exists and refresh not requested
    if cache_path and Path(cache_path).exists() and not force_refresh:
        logger.info(f"Loading {ticker} from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"  Loaded {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    logger.info(f"Downloading {ticker} from Yahoo Finance ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    _validate_ohlcv(df, ticker)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"  Cached to {cache_path}")

    logger.info(f"  Downloaded {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load_spy_and_vix(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: load both SPY and VIX using config values.

    Returns:
        (spy_df, vix_df) — both as clean OHLCV DataFrames
    """
    spy = download_ticker(
        ticker=cfg.data.ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_path=cfg.data.raw_path,
        force_refresh=force_refresh,
    )
    vix = download_ticker(
        ticker=cfg.data.vix_ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_path=cfg.data.vix_path,
        force_refresh=force_refresh,
    )
    return spy, vix


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> None:
    """
    Basic data quality checks. Raises ValueError on critical issues,
    logs warnings for minor ones.
    """
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if len(df) < 252:
        raise ValueError(f"{ticker}: Only {len(df)} rows — need at least 1 year of data")

    zero_vol = (df.get("Volume", pd.Series(dtype=float)) == 0).sum()
    if zero_vol > 0:
        logger.warning(f"{ticker}: {zero_vol} days with zero volume")

    neg_price = (df["Close"] <= 0).sum()
    if neg_price > 0:
        raise ValueError(f"{ticker}: {neg_price} non-positive Close prices detected")

    # Check for large gaps (>5 trading days) — may indicate data corruption
    date_diffs = pd.Series(df.index).diff().dt.days.dropna()
    large_gaps = date_diffs[date_diffs > 5]
    if len(large_gaps) > 0:
        logger.warning(f"{ticker}: {len(large_gaps)} gaps > 5 calendar days in date index")


def load_market_data(force_refresh: bool = False) -> dict:
    """
    Load all required market data using config values.
    Returns a dict so callers access by name — easy to extend with new sources.

    Returns:
        {
          "price" : NSEI OHLCV,
          "vix"   : India VIX,
          "usdinr": USD/INR rate  (if configured),
          "crude" : Brent crude   (if configured),
        }
    """
    data = {}

    data["price"] = download_ticker(
        ticker=cfg.data.ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_path=cfg.data.raw_path,
        force_refresh=force_refresh,
    )
    data["vix"] = download_ticker(
        ticker=cfg.data.vix_ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_path=cfg.data.vix_path,
        force_refresh=force_refresh,
    )

    # India-specific macro — gracefully skip if ticker not in config
    for key, ticker_attr, path_attr in [
        ("usdinr", "usdinr_ticker", "usdinr_path"),
        ("crude",  "crude_ticker",  "crude_path"),
    ]:
        ticker = cfg.data.get(ticker_attr)
        path   = cfg.data.get(path_attr)
        if ticker:
            try:
                data[key] = download_ticker(
                    ticker=ticker,
                    start=cfg.data.start_date,
                    end=cfg.data.end_date,
                    cache_path=path,
                    force_refresh=force_refresh,
                )
            except Exception as e:
                logger.warning(f"Could not load {key} ({ticker}): {e} — skipping")

    logger.info(f"Loaded datasets: {list(data.keys())}")
    return data