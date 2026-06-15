"""
Live market data fetcher using yfinance.

What we get for free:
  - SPX spot price and options chain (IV surface)
  - VIX spot level (used as proxy for nearest VIX futures)
  - VIX options chain (when available)

What requires a paid feed:
  - VIX futures quotes beyond the spot level
  - Intraday tick data
  - Pre-2020 options history

All timestamps are returned in US/Eastern.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional


# ── Spot levels ───────────────────────────────────────────────────────────────

def get_spot_levels() -> dict:
    """
    Fetch current SPX and VIX levels.

    Returns
    -------
    dict with keys: 'spx', 'vix', 'timestamp'
    """
    tickers = yf.download(["^SPX", "^VIX"], period="1d",
                           interval="1m", progress=False)
    spx = float(tickers["Close"]["^SPX"].dropna().iloc[-1])
    vix = float(tickers["Close"]["^VIX"].dropna().iloc[-1])
    return {
        "spx": spx,
        "vix": vix,
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ── SPX options ───────────────────────────────────────────────────────────────

def get_spx_smile(
    n_expiries: int = 2,
    min_volume: int = 0,
    moneyness_range: tuple = (0.80, 1.20),
) -> list[dict]:
    """
    Fetch SPX implied volatility smiles for the nearest n_expiries expiry dates.

    Yahoo Finance returns implied volatility already computed (mid-price
    Black-Scholes inversion). We filter for reasonable moneyness and
    sufficient liquidity (open interest > 0).

    Parameters
    ----------
    n_expiries       : number of expiry dates to fetch
    min_volume       : minimum open interest to include a strike
    moneyness_range  : (lo, hi) as fraction of spot, e.g. (0.80, 1.20)

    Returns
    -------
    list of dicts, one per expiry, each with:
        'expiry'    : expiry date string
        'T'         : time to expiry in years
        'strikes'   : array of strike prices
        'ivs'       : array of implied vols (from Yahoo Finance)
        'bid'       : array of bid prices
        'ask'       : array of ask prices
        'forward'   : forward price (spot, assuming r=0)
    """
    spx = yf.Ticker("^SPX")
    spot = spx.history(period="1d")["Close"].iloc[-1]
    today = pd.Timestamp.now(tz="America/New_York").normalize()

    all_expiries = spx.options
    if not all_expiries:
        return []

    results = []
    for exp_str in all_expiries[:n_expiries]:
        exp_date = pd.Timestamp(exp_str, tz="America/New_York")
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        chain = spx.option_chain(exp_str)
        calls = chain.calls.copy()

        # Filter: moneyness window and non-zero open interest
        lo, hi = moneyness_range
        calls = calls[
            (calls["strike"] >= spot * lo) &
            (calls["strike"] <= spot * hi) &
            (calls["openInterest"] > min_volume) &
            (calls["impliedVolatility"] > 0.01) &
            (calls["impliedVolatility"] < 5.0)
        ].copy()

        if len(calls) < 3:
            continue

        calls = calls.sort_values("strike")

        results.append({
            "expiry":  exp_str,
            "T":       T,
            "strikes": calls["strike"].values,
            "ivs":     calls["impliedVolatility"].values,
            "bid":     calls["bid"].values,
            "ask":     calls["ask"].values,
            "forward": float(spot),
        })

    return results


# ── VIX futures (proxy) ───────────────────────────────────────────────────────

def get_vix_futures_proxy() -> list[dict]:
    """
    Return VIX spot as a proxy for the front-month VIX futures price.

    Limitation: the true futures price includes a basis (futures trade at a
    premium to spot during normal conditions — contango). For calibration,
    a paid feed (Polygon.io, CBOE DataShop) would give the actual futures
    price per expiry. Here we use the VIX spot as a best free approximation.

    Returns a list of one entry matching the structure expected by calibration.
    """
    vix = yf.Ticker("^VIX")
    spot = float(vix.history(period="1d")["Close"].iloc[-1])
    today = pd.Timestamp.now(tz="America/New_York").normalize()

    # VIX futures expire on the Wednesday 30 days before the next SPX expiry.
    # Approximate: use the nearest monthly Wednesday ~28 days out.
    import pandas.tseries.offsets as offsets
    approx_expiry = today + pd.Timedelta(days=28)

    # Snap to nearest Wednesday
    days_to_wed = (2 - approx_expiry.weekday()) % 7
    approx_expiry = approx_expiry + pd.Timedelta(days=days_to_wed)

    T = (approx_expiry - today).days / 365.0

    return [{
        "expiry":       approx_expiry.strftime("%Y-%m-%d"),
        "T":            T,
        "futures_price": spot,     # VIX spot used as futures proxy
        "is_proxy":     True,      # flag so dashboard can warn user
    }]


# ── VIX options ───────────────────────────────────────────────────────────────

def get_vix_smile(
    n_expiries: int = 1,
    moneyness_range: tuple = (0.70, 2.00),
) -> list[dict]:
    """
    Fetch VIX options implied volatility smile.

    VIX options are quoted in implied vol of the VIX itself — a vol-of-vol.
    The moneyness is expressed as K / F_VIX where F_VIX is the futures price
    for that expiry (approximated here by VIX spot).
    """
    vix_ticker = yf.Ticker("^VIX")
    vix_spot   = float(vix_ticker.history(period="1d")["Close"].iloc[-1])
    today      = pd.Timestamp.now(tz="America/New_York").normalize()

    all_expiries = vix_ticker.options
    if not all_expiries:
        return []

    results = []
    for exp_str in all_expiries[:n_expiries]:
        exp_date = pd.Timestamp(exp_str, tz="America/New_York")
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        chain = vix_ticker.option_chain(exp_str)
        calls = chain.calls.copy()

        lo, hi = moneyness_range
        calls = calls[
            (calls["strike"] >= vix_spot * lo) &
            (calls["strike"] <= vix_spot * hi) &
            (calls["openInterest"] > 0) &
            (calls["impliedVolatility"] > 0.01)
        ].copy()

        if len(calls) < 3:
            continue

        calls = calls.sort_values("strike")
        moneyness = calls["strike"].values / vix_spot

        results.append({
            "expiry":     exp_str,
            "T":          T,
            "strikes":    calls["strike"].values,
            "moneyness":  moneyness,
            "ivs":        calls["impliedVolatility"].values,
            "bid":        calls["bid"].values,
            "ask":        calls["ask"].values,
            "futures_px": vix_spot,
        })

    return results


# ── Combined snapshot ─────────────────────────────────────────────────────────

def get_market_snapshot(
    n_spx_expiries: int = 2,
    n_vix_expiries: int = 1,
) -> dict:
    """
    Single call that returns everything needed for calibration and display.
    """
    levels    = get_spot_levels()
    spx_smile = get_spx_smile(n_expiries=n_spx_expiries)
    vix_fut   = get_vix_futures_proxy()
    vix_smile = get_vix_smile(n_expiries=n_vix_expiries)

    return {
        "levels":     levels,
        "spx_smiles": spx_smile,
        "vix_futures": vix_fut,
        "vix_smiles": vix_smile,
        "fetched_at": levels["timestamp"],
    }
