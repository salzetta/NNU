"""
VIX pricing via Least Squares Monte Carlo (LSMC).

VIX definition (from equation 3 in the paper, no-jump case):

    VIX²_T = (1/Δ) ∫_{T}^{T+Δ} E[σ²_u | F_T] du

where Δ = 30/365 (30-day horizon).

Nested MC computes this by:
  - Outer sim: n_out paths from t=0 to T  →  terminal states x_T
  - Inner sim: for each x_T, run n_inner paths from T to T+Δ, average σ²

LSMC accelerates this by:
  1. Running inner sim only on a subsample of n_sub outer paths.
  2. Fitting a degree-3 polynomial: VIX_T ≈ poly(R1_0, R1_1, R2_0, R2_1, σ)
  3. Evaluating the polynomial on ALL outer paths.
"""

import numpy as np
from itertools import combinations_with_replacement
from src.model.pdv_model import PDVParams, simulate, compute_sigma

DELTA = 30 / 365   # VIX 30-day horizon in years


# ── Polynomial feature utilities ──────────────────────────────────────────────

def _poly_features(X: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Build polynomial features up to a given degree, including a bias column.

    Parameters
    ----------
    X      : (n_samples, n_features)
    degree : maximum total degree of monomials

    Returns
    -------
    (n_samples, n_poly_features) — includes the bias (intercept) column
    """
    n_samples, n_features = X.shape
    cols = [np.ones((n_samples, 1))]    # bias
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            cols.append(np.prod(X[:, combo], axis=1, keepdims=True))
    return np.hstack(cols)


def _ridge_fit(Phi: np.ndarray, y: np.ndarray,
               alpha: float) -> np.ndarray:
    """
    Fit ridge regression: minimise ||y - Phi @ beta||² + alpha * ||beta||².

    Closed-form solution: beta = (Phi'Phi + alpha*I)^{-1} Phi'y
    """
    A = Phi.T @ Phi
    A[np.diag_indices_from(A)] += alpha
    b = Phi.T @ y
    return np.linalg.solve(A, b)


# ── Inner simulation ───────────────────────────────────────────────────────────

def _inner_vix(
    params: PDVParams,
    R1_0: np.ndarray,
    R1_1: np.ndarray,
    R2_0: np.ndarray,
    R2_1: np.ndarray,
    n_inner: int,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Estimate VIX_T for each of n_sub terminal states via inner Monte Carlo.

    For every terminal state, we fan out n_inner paths over Δ = 30/365 years
    and average σ² to approximate E[σ²|F_T].

    Parameters
    ----------
    R1_0, R1_1, R2_0, R2_1 : arrays of shape (n_sub,) — terminal factor values

    Returns
    -------
    vix : shape (n_sub,)
    """
    if rng is None:
        rng = np.random.default_rng()

    p = params
    n_sub      = len(R1_0)
    n_steps    = max(1, round(DELTA / dt))
    sqrt_dt    = np.sqrt(dt)

    # Broadcast each outer state across n_inner inner paths: shape (n_sub, n_inner)
    r1_0 = np.tile(R1_0[:, None], (1, n_inner))
    r1_1 = np.tile(R1_1[:, None], (1, n_inner))
    r2_0 = np.tile(R2_0[:, None], (1, n_inner))
    r2_1 = np.tile(R2_1[:, None], (1, n_inner))

    r1  = (1 - p.theta1) * r1_0 + p.theta1 * r1_1
    r2  = (1 - p.theta2) * r2_0 + p.theta2 * r2_1
    sig = compute_sigma(r1, r2, p)

    sigma_sq_sum = np.zeros((n_sub, n_inner))

    for _ in range(n_steps):
        sigma_sq_sum += sig**2

        dW   = rng.standard_normal((n_sub, n_inner)) * sqrt_dt
        r1_0 = r1_0 + p.lam1_0 * (sig * dW - r1_0 * dt)
        r1_1 = r1_1 + p.lam1_1 * (sig * dW - r1_1 * dt)
        r2_0 = r2_0 + p.lam2_0 * (sig**2 - r2_0) * dt
        r2_1 = r2_1 + p.lam2_1 * (sig**2 - r2_1) * dt

        r1  = (1 - p.theta1) * r1_0 + p.theta1 * r1_1
        r2  = (1 - p.theta2) * r2_0 + p.theta2 * r2_1
        sig = compute_sigma(r1, r2, p)

    # Average over inner paths and time steps → VIX²_T
    vix2 = (sigma_sq_sum / n_steps).mean(axis=1)
    return np.sqrt(np.maximum(vix2, 0.0))


# ── LSMC pricer ───────────────────────────────────────────────────────────────

def price_vix_lsmc(
    params: PDVParams,
    T_vix: float,
    strikes: np.ndarray,
    n_out: int = 4096,
    n_sub: int = 256,
    n_inner: int = 512,
    poly_degree: int = 3,
    ridge_alpha: float = 1e-4,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> dict:
    """
    Price VIX futures and call options via LSMC.

    Parameters
    ----------
    T_vix       : VIX expiry in years (e.g. 28/365)
    strikes     : array of call strikes (as fractions of expected VIX level)
    n_out       : total outer paths
    n_sub       : subsample size for inner simulation (n_sub <= n_out)
    n_inner     : inner paths per subsample state
    poly_degree : polynomial degree for regression (paper uses 3)
    ridge_alpha : ridge penalty (prevents overfitting the subsample)

    Returns
    -------
    dict with keys:
        'future'     : VIX futures price  F_VIX = E[VIX_T]
        'calls'      : call option prices C_VIX(K), shape (n_strikes,)
        'vix_paths'  : estimated VIX_T for all outer paths, shape (n_out,)
        'vix_sub'    : VIX_T from inner sim on subsample,   shape (n_sub,)
        'r2_train'   : R² of the polynomial fit on subsample
    """
    assert n_sub <= n_out, "n_sub must be <= n_out"
    rng = np.random.default_rng(seed)

    # ── Step 1: outer simulation ──────────────────────────────────────────────
    n_steps_out = max(1, round(T_vix / dt))
    outer = simulate(params, n_paths=n_out, n_steps=n_steps_out,
                     dt=dt, seed=int(rng.integers(0, 2**31)),
                     store_paths=False)

    # Terminal factor values for all outer paths
    R1_0_all = outer['R1_0']
    R1_1_all = outer['R1_1']
    R2_0_all = outer['R2_0']
    R2_1_all = outer['R2_1']
    sig_all  = outer['vol']

    # ── Step 2: subsample and run inner simulation ────────────────────────────
    idx_sub = rng.choice(n_out, size=n_sub, replace=False)

    vix_sub = _inner_vix(
        params,
        R1_0_all[idx_sub], R1_1_all[idx_sub],
        R2_0_all[idx_sub], R2_1_all[idx_sub],
        n_inner=n_inner, dt=dt, rng=rng,
    )

    # ── Step 3: fit polynomial regression on the subsample ───────────────────
    # Features: the 5 state variables at time T for the subsample
    X_sub = np.column_stack([
        R1_0_all[idx_sub], R1_1_all[idx_sub],
        R2_0_all[idx_sub], R2_1_all[idx_sub],
        sig_all[idx_sub],
    ])
    Phi_sub = _poly_features(X_sub, degree=poly_degree)
    beta    = _ridge_fit(Phi_sub, vix_sub, alpha=ridge_alpha)

    # R² on the subsample (diagnostic for fit quality)
    vix_sub_hat = Phi_sub @ beta
    ss_res = np.sum((vix_sub - vix_sub_hat)**2)
    ss_tot = np.sum((vix_sub - vix_sub.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # ── Step 4: generalise to all outer paths ─────────────────────────────────
    X_all   = np.column_stack([R1_0_all, R1_1_all, R2_0_all, R2_1_all, sig_all])
    Phi_all = _poly_features(X_all, degree=poly_degree)
    vix_all = Phi_all @ beta
    vix_all = np.maximum(vix_all, 0.0)   # VIX is non-negative

    # ── Step 5: price derivatives ─────────────────────────────────────────────
    future = vix_all.mean()
    calls  = np.array([np.maximum(vix_all - K, 0.0).mean() for K in strikes])

    return {
        'future':    future,
        'calls':     calls,
        'vix_paths': vix_all,
        'vix_sub':   vix_sub,
        'r2_train':  r2,
    }
