"""
Black-Scholes pricing and implied volatility inversion.

We work with forward prices and zero interest rates, consistent with the
paper's normalised setup (S0 = 1, r = 0, so the forward price F = S0).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call(F: float, K: float, T: float, vol: float) -> float:
    """
    Black-Scholes call price given forward price F, strike K,
    maturity T (years), and volatility vol.

    With r=0: C = F*N(d1) - K*N(d2)
    """
    if vol <= 0 or T <= 0:
        return max(F - K, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def bs_call_vectorised(F: float, K: np.ndarray, T: float,
                       vol: np.ndarray) -> np.ndarray:
    """Vectorised version of bs_call over arrays of K and vol."""
    vol = np.asarray(vol)
    K   = np.asarray(K)
    sqrt_T = np.sqrt(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol(price: float, F: float, K: float, T: float,
                vol_lo: float = 1e-4, vol_hi: float = 10.0) -> float:
    """
    Invert Black-Scholes to find the implied volatility.

    Uses Brent's method — guaranteed to converge on a valid bracket.
    Returns NaN when the price is below intrinsic value (no solution exists).
    """
    intrinsic = max(F - K, 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan

    objective = lambda v: bs_call(F, K, T, v) - price

    # Check the bracket contains a sign change before calling brentq
    if objective(vol_lo) * objective(vol_hi) > 0:
        return np.nan

    try:
        return brentq(objective, vol_lo, vol_hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan


def implied_vol_surface(prices: np.ndarray, F: float,
                        strikes: np.ndarray, T: float) -> np.ndarray:
    """
    Compute implied vols for a vector of prices at one maturity.
    Returns an array of the same length, with NaN for unsolvable entries.
    """
    return np.array([implied_vol(p, F, K, T) for p, K in zip(prices, strikes)])
