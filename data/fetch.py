"""
Data fetching utilities.

ETF prices: massive API (adjusted close prices, close-to-close simple returns).
Risk-free rate: FRED Federal Funds Effective Rate (DFF), converted to daily
simple return as r_rf_k = (annual_rate_pct / 100) / 252.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from massive import RESTClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASSIVE_API_KEY: str = os.getenv("MASSIVE_TOKEN", "")
FRED_SERIES_FF: str = "DFF"  # Federal Funds Effective Rate (daily, % per annum)


# ---------------------------------------------------------------------------
# ETF price data
# ---------------------------------------------------------------------------

def fetch_etf_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for a list of ETF tickers via massive API.

    Parameters
    ----------
    tickers : list of str
        ETF ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end : str
        End date in 'YYYY-MM-DD' format (inclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker containing
        adjusted close prices.  Missing values are left as NaN.
    """
    if not MASSIVE_API_KEY:
        raise EnvironmentError(
            "MASSIVE_TOKEN environment variable is not set. "
            "Please export MASSIVE_TOKEN=<your_api_key>."
        )

    client = RESTClient(api_key=MASSIVE_API_KEY)
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        logger.info("Fetching %s from %s to %s", ticker, start, end)
        aggs = []
        try:
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start,
                to=end,
                limit=50000,
                adjusted=True,
            ):
                aggs.append(a)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            frames[ticker] = pd.Series(dtype=float, name=ticker)
            continue

        if not aggs:
            logger.warning("No data returned for %s", ticker)
            frames[ticker] = pd.Series(dtype=float, name=ticker)
            continue

        dates = pd.to_datetime([a.timestamp for a in aggs], unit="ms", utc=True).tz_convert(None)
        closes = [a.close for a in aggs]
        s = pd.Series(closes, index=dates, name=ticker)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        frames[ticker] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.sort_index()
    # Filter to requested date range
    df = df.loc[start:end]
    return df


def fetch_single_etf_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch daily adjusted close prices for a single ETF ticker.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol.
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.Series
        Adjusted close prices with DatetimeIndex.
    """
    df = fetch_etf_prices([ticker], start, end)
    if ticker not in df.columns or df[ticker].dropna().empty:
        raise ValueError(f"No price data returned for {ticker} between {start} and {end}.")
    return df[ticker].dropna()


# ---------------------------------------------------------------------------
# Federal Funds Effective Rate
# ---------------------------------------------------------------------------

def fetch_fed_funds_rate(start: str, end: str) -> pd.Series:
    """
    Fetch the Federal Funds Effective Rate (DFF) from FRED.

    Returns a daily series of the annualized rate in percent (e.g., 5.33 means 5.33%).
    Missing values (weekends/holidays) are forward-filled.

    Parameters
    ----------
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.Series
        Daily Federal Funds Effective Rate (% per annum), forward-filled.
    """
    try:
        ff = web.DataReader(FRED_SERIES_FF, "fred", start, end)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Federal Funds rate from FRED: {exc}") from exc

    ff = ff[FRED_SERIES_FF].rename("fed_funds_rate")
    ff = ff.ffill()
    ff.index = pd.to_datetime(ff.index)
    ff.index.name = "date"
    return ff


def compute_daily_rf_return(ff_rate: pd.Series) -> pd.Series:
    """
    Convert annualized Federal Funds rate (% per annum) to a daily simple return.

    Convention: r_rf_k = (annual_rate_pct / 100) / 252.
    This is a simple linear approximation consistent with the paper's intent.

    Parameters
    ----------
    ff_rate : pd.Series
        Annualized Federal Funds rate in percent (e.g., 5.33 for 5.33%).

    Returns
    -------
    pd.Series
        Daily simple risk-free return.
    """
    return (ff_rate / 100.0) / 252.0


# ---------------------------------------------------------------------------
# Aligned return series builder
# ---------------------------------------------------------------------------

def build_aligned_returns(
    prices: pd.Series,
    ff_rate: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Build aligned daily simple return series for a risky asset and cash.

    The risky return is r_k = P_k / P_{k-1} - 1 (close-to-close simple return).
    The cash return is aligned to the ETF trading calendar by forward-filling
    the Fed Funds rate to match ETF trading days, then converting to daily return.

    Parameters
    ----------
    prices : pd.Series
        Adjusted close prices for the risky asset.
    ff_rate : pd.Series
        Daily Federal Funds Effective Rate (% per annum), possibly on all calendar days.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (risky_returns, cash_returns) both indexed on the ETF trading calendar
        starting from the second available price date.
    """
    prices = prices.dropna().sort_index()
    risky_returns = prices.pct_change().dropna()

    # Align Fed Funds to ETF trading calendar
    ff_aligned = ff_rate.reindex(risky_returns.index, method="ffill")
    # Fill any remaining NaN at the start with 0
    ff_aligned = ff_aligned.fillna(0.0)
    cash_returns = compute_daily_rf_return(ff_aligned)

    return risky_returns, cash_returns
