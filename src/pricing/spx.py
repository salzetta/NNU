"""
SPX option pricing via Monte Carlo + Black-Scholes implied volatility.

Workflow:
  1. Simulate n_paths asset price paths to maturity T.
  2. Compute European call payoffs: max(S_T - K, 0).
  3. Average payoffs → Monte Carlo price.
  4. Invert Black-Scholes → implied volatility.
"""

import numpy as np
from src.model.pdv_model import PDVParams, simulate
from src.pricing.black_scholes import implied_vol_surface


def price_spx_calls(
    params: PDVParams,
    T: float,
    strikes: np.ndarray,
    n_paths: int = 10_000,
    S0: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Price European call options on the SPX at a single maturity T.

    Parameters
    ----------
    params  : PDVParams
    T       : time to maturity in years
    strikes : array of strike prices (as fractions of S0, e.g. 0.9, 1.0, 1.1)
    n_paths : number of Monte Carlo paths
    S0      : initial asset price (default 1.0)

    Returns
    -------
    dict with keys:
        'prices' : Monte Carlo call prices,    shape (n_strikes,)
        'ivs'    : Black-Scholes implied vols, shape (n_strikes,)
        'S_T'    : simulated terminal prices,  shape (n_paths,)
        'se'     : standard error per strike,  shape (n_strikes,)
    """
    dt = 1 / 252
    n_steps = max(1, round(T / dt))

    result = simulate(params, n_paths=n_paths, n_steps=n_steps,
                      dt=dt, S0=S0, seed=seed, store_paths=False)
    S_T = result['S']

    strikes = np.asarray(strikes)
    prices = np.empty(len(strikes))
    se     = np.empty(len(strikes))

    for i, K in enumerate(strikes):
        payoffs    = np.maximum(S_T - K, 0.0)
        prices[i]  = payoffs.mean()
        se[i]      = payoffs.std() / np.sqrt(n_paths)

    # Forward price = S0 (r = 0)
    ivs = implied_vol_surface(prices, F=S0, strikes=strikes, T=T)

    return {'prices': prices, 'ivs': ivs, 'S_T': S_T, 'se': se}


def price_spx_surface(
    params: PDVParams,
    maturities: np.ndarray,
    strikes_per_maturity: list[np.ndarray],
    n_paths: int = 20_000,
    S0: float = 1.0,
    seed: int | None = None,
) -> list[dict]:
    """
    Price calls across multiple maturities using a single set of simulated paths.

    Because each maturity needs paths up to a different horizon, we simulate
    up to the longest maturity and read off intermediate terminal prices.
    Each maturity slice gets the same Brownian paths, so the surface is
    internally consistent.

    Returns a list of dicts (one per maturity), each with the same keys as
    price_spx_calls.
    """
    dt = 1 / 252
    T_max   = float(np.max(maturities))
    n_steps = max(1, round(T_max / dt))

    rng = np.random.default_rng(seed)
    # Full path simulation — store_paths=True needed here
    from src.model.pdv_model import simulate as _simulate
    paths = _simulate(params, n_paths=n_paths, n_steps=n_steps,
                      dt=dt, S0=S0, seed=rng.integers(0, 2**31),
                      store_paths=True)
    S_paths = paths['S']   # (n_paths, n_steps+1)

    results = []
    for T, strikes in zip(maturities, strikes_per_maturity):
        step_idx = min(round(T / dt), n_steps)
        S_T      = S_paths[:, step_idx]
        strikes  = np.asarray(strikes)

        prices = np.empty(len(strikes))
        se     = np.empty(len(strikes))
        for i, K in enumerate(strikes):
            payoffs   = np.maximum(S_T - K, 0.0)
            prices[i] = payoffs.mean()
            se[i]     = payoffs.std() / np.sqrt(n_paths)

        ivs = implied_vol_surface(prices, F=S0, strikes=strikes, T=T)
        results.append({'T': T, 'strikes': strikes,
                        'prices': prices, 'ivs': ivs, 'se': se})

    return results
