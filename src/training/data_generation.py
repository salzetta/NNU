"""
Training data generation for the SPX and VIX neural networks.

For each sampled parameter combination θ, we:
  - Draw random maturities and strikes (pointwise approach)
  - Price SPX calls → convert to implied vols
  - Price VIX futures and calls via LSMC
  - Store each (θ, T, K) → target as a flat row

The result is two datasets:
  spx_data : (n_points, 16) inputs  →  (n_points, 1) implied vols
  vix_data : (n_points, 16) inputs  →  (n_points, 2) [futures price, call price]
"""

import numpy as np
from dataclasses import astuple
from tqdm import tqdm

from src.model.pdv_model import PDVParams
from src.pricing.spx import price_spx_calls
from src.pricing.vix import price_vix_lsmc, DELTA

# ── Parameter bounds from Table A of the paper ────────────────────────────────

PARAM_BOUNDS = {
    'beta0':  (0.00,  0.85),
    'beta1':  (-0.30, -0.10),
    'beta2':  (0.35,  0.95),
    'beta12': (0.05,  0.40),
    'lam1_0': (10.0,  65.0),
    'lam1_1': (0.0,   35.0),
    'theta1': (0.0,   1.0),
    'lam2_0': (0.0,   50.0),
    'lam2_1': (0.0,   15.0),
    'theta2': (0.0,   1.0),
    'R1_0_0': (-1.62, 0.88),
    'R1_1_0': (-1.05, 0.71),
    'R2_0_0': (0.0,   0.11),
    'R2_1_0': (0.0,   0.11),
}

# Maturity buckets (years) — from the paper's SPX training setup
SPX_MATURITY_BUCKETS = [
    (6/365,   1/12),
    (1/12,    2/12),
    (2/12,    3/12),
    (3/12,    4/12),
    (4/12,    5/12),
    (5/12,    6/12),
    (6/12,    8/12),
    (8/12,   10/12),
    (10/12,  11/12),
    (11/12,   1.0),
    (1.0,    13/12),
]

# VIX maturity buckets (years) — from the paper
VIX_MATURITY_BUCKETS = [
    (6/365,  18/365),
    (18/365, 30/365),
]

# Strike range parameters for SPX (from the paper)
SPX_L, SPX_U = 0.55, 0.30   # K in [S0*(1 - l*sqrt(T)), S0*(1 + u*sqrt(T))]

# VIX moneyness range: K/F in [0.82, 2.36] (from the paper)
VIX_MONEYNESS_RANGE = (0.82, 2.36)


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_params(rng: np.random.Generator) -> PDVParams | None:
    """
    Draw one parameter set uniformly from the hypercube in Table A.
    Returns None if the λ ordering constraint is violated.
    """
    lo = {k: v[0] for k, v in PARAM_BOUNDS.items()}
    hi = {k: v[1] for k, v in PARAM_BOUNDS.items()}

    vals = {k: rng.uniform(lo[k], hi[k]) for k in PARAM_BOUNDS}

    # Enforce fast > slow for both R1 and R2 factors
    if vals['lam1_0'] <= vals['lam1_1']:
        return None
    if vals['lam2_0'] <= vals['lam2_1']:
        return None

    return PDVParams(**vals)


def _is_realistic(params: PDVParams) -> bool:
    """
    Quick check: initial vol must be in a reasonable range, and a short
    forward simulation must not blow up.
    """
    from src.model.pdv_model import compute_sigma, simulate
    R1 = (1 - params.theta1) * params.R1_0_0 + params.theta1 * params.R1_1_0
    R2 = (1 - params.theta2) * params.R2_0_0 + params.theta2 * params.R2_1_0
    sig0 = compute_sigma(np.array([R1]), np.array([R2]), params)[0]
    if not (0.02 < sig0 < 2.0):
        return False
    # Short simulation check: ensure paths stay finite over 30 days
    try:
        out = simulate(params, n_paths=64, n_steps=30, seed=0, store_paths=False)
        if not np.all(np.isfinite(out['S'])) or not np.all(np.isfinite(out['vol'])):
            return False
        if out['vol'].max() > 4.0:
            return False
    except Exception:
        return False
    return True


# ── SPX training data ─────────────────────────────────────────────────────────

def generate_spx_data(
    n_params: int,
    n_maturities_per_param: int = 11,
    n_strikes_per_maturity: int = 8,
    mc_paths: int = 8_000,
    seed: int = 0,
    realistic_fraction: float = 0.8,
) -> dict:
    """
    Generate SPX implied volatility training data.

    For each valid parameter set, sample random maturities (one per bucket)
    and random strikes around ATM, then price via Monte Carlo.

    Parameters
    ----------
    n_params                : number of parameter combinations to attempt
    n_maturities_per_param  : how many maturities to sample (≤ 11 buckets)
    n_strikes_per_maturity  : strikes sampled per maturity
    mc_paths                : Monte Carlo paths per parameter set
    realistic_fraction      : fraction of realistic params to keep
                              (rest kept for coverage, per the paper)

    Returns
    -------
    dict with:
        'inputs'  : (N, 16) array — (θ, T, K)
        'targets' : (N, 1)  array — implied vol
    """
    rng = np.random.default_rng(seed)
    n_buckets = len(SPX_MATURITY_BUCKETS)
    n_mat = min(n_maturities_per_param, n_buckets)

    all_inputs  = []
    all_targets = []
    skipped     = 0

    for _ in tqdm(range(n_params), desc='SPX data generation'):
        params = None
        for _ in range(20):          # retry up to 20 times for valid params
            params = sample_params(rng)
            if params is not None:
                break
        if params is None:
            skipped += 1
            continue

        realistic = _is_realistic(params)
        if not realistic and rng.random() > (1 - realistic_fraction):
            skipped += 1
            continue

        # Sample one maturity per bucket (randomly within bucket)
        bucket_indices = rng.choice(n_buckets, size=n_mat, replace=False)
        maturities = np.array([
            rng.uniform(*SPX_MATURITY_BUCKETS[i]) for i in bucket_indices
        ])

        theta_vec = np.array([
            params.beta0, params.beta1, params.beta2, params.beta12,
            params.lam1_0, params.lam1_1, params.theta1,
            params.lam2_0, params.lam2_1, params.theta2,
            params.R1_0_0, params.R1_1_0, params.R2_0_0, params.R2_1_0,
        ])

        for T in maturities:
            # Random strikes around ATM: K in [1 - l*sqrt(T), 1 + u*sqrt(T)]
            K_lo = max(1.0 - SPX_L * np.sqrt(T), 0.3)
            K_hi = 1.0 + SPX_U * np.sqrt(T)
            strikes = rng.uniform(K_lo, K_hi, size=n_strikes_per_maturity)

            try:
                result = price_spx_calls(
                    params, T=T, strikes=strikes,
                    n_paths=mc_paths,
                    seed=int(rng.integers(0, 2**31)),
                )
            except Exception:
                continue

            for K, iv in zip(strikes, result['ivs']):
                if np.isnan(iv) or iv <= 0 or iv > 5.0:
                    continue
                row = np.append(theta_vec, [T, K])
                all_inputs.append(row)
                all_targets.append(iv)

    print(f"SPX: generated {len(all_targets):,} points "
          f"({skipped} param sets skipped)")

    return {
        'inputs':  np.array(all_inputs,  dtype=np.float32),
        'targets': np.array(all_targets, dtype=np.float32).reshape(-1, 1),
    }


# ── VIX training data ─────────────────────────────────────────────────────────

def generate_vix_data(
    n_params: int,
    n_strikes_per_maturity: int = 20,
    n_out: int = 2048,
    n_sub: int = 128,
    n_inner: int = 256,
    seed: int = 1,
) -> dict:
    """
    Generate VIX futures and call option training data via LSMC.

    Each training point maps (θ, T, m) → (F_VIX, C_VIX)
    where m = K / F_VIX is moneyness relative to the futures price.

    Returns
    -------
    dict with:
        'inputs'  : (N, 16) array — (θ, T, m)
        'targets' : (N, 2)  array — [futures price, call price]
    """
    rng = np.random.default_rng(seed)

    all_inputs  = []
    all_targets = []
    skipped     = 0

    # Moneyness grid: 4 strikes in [0.82,1.00], 7 in [1.00,1.40], 9 in [1.40,2.36]
    m_lo  = np.linspace(VIX_MONEYNESS_RANGE[0], 1.00, 4)
    m_mid = np.linspace(1.00, 1.40, 7)
    m_hi  = np.linspace(1.40, VIX_MONEYNESS_RANGE[1], 9)
    moneyness_grid = np.unique(np.concatenate([m_lo, m_mid, m_hi]))

    for _ in tqdm(range(n_params), desc='VIX data generation'):
        params = None
        for _ in range(20):
            params = sample_params(rng)
            if params is not None:
                break
        if params is None:
            skipped += 1
            continue
        if not _is_realistic(params):
            skipped += 1
            continue

        theta_vec = np.array([
            params.beta0, params.beta1, params.beta2, params.beta12,
            params.lam1_0, params.lam1_1, params.theta1,
            params.lam2_0, params.lam2_1, params.theta2,
            params.R1_0_0, params.R1_1_0, params.R2_0_0, params.R2_1_0,
        ])

        for bucket in VIX_MATURITY_BUCKETS:
            T = rng.uniform(*bucket)

            # Sample a random subset of moneyness values
            n_sub_strikes = min(n_strikes_per_maturity, len(moneyness_grid))
            sel = rng.choice(len(moneyness_grid), size=n_sub_strikes, replace=False)
            m_sel = moneyness_grid[sel]

            try:
                result = price_vix_lsmc(
                    params, T_vix=T,
                    strikes=np.array([0.0]),   # placeholder, we'll compute per-m below
                    n_out=n_out, n_sub=n_sub, n_inner=n_inner,
                    seed=int(rng.integers(0, 2**31)),
                )
            except Exception:
                continue

            F = result['future']
            if F <= 0 or np.isnan(F):
                continue

            # Re-price calls at the sampled moneyness levels
            vix_paths = result['vix_paths']
            for m in m_sel:
                K = m * F
                call_price = np.maximum(vix_paths - K, 0.0).mean()
                if np.isnan(call_price):
                    continue
                row = np.append(theta_vec, [T, m])
                all_inputs.append(row)
                all_targets.append([F, call_price])

    print(f"VIX: generated {len(all_targets):,} points "
          f"({skipped} param sets skipped)")

    return {
        'inputs':  np.array(all_inputs,  dtype=np.float32),
        'targets': np.array(all_targets, dtype=np.float32),
    }
