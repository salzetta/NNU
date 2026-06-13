"""
Phase 4 — Joint SPX/VIX calibration.

The calibrator finds the 10 model parameters θ^M that minimise the joint
loss function (equation 5 of the paper) against live market data.

Key ideas:
  - The trained neural networks replace Monte Carlo pricing. Evaluating
    the model is a single matrix-vector product — fast enough for real-time
    gradient descent.
  - Only θ^M = (β₀, β₁, β₂, β₁₂, λ₁₀, λ₁₁, θ₁, λ₂₀, λ₂₁, θ₂) is
    optimised. The initial factor values θ^R = (R₁₀₀, R₁₁₀, R₂₀₀, R₂₁₀)
    are computed deterministically from θ^M and the observed SPX price
    history at each gradient step.
  - Constraints are handled via sigmoid reparameterisation so gradient
    descent can run freely over unconstrained parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from dataclasses import dataclass

from src.model.pdv_model import PDVParams
from src.training.networks import PDVNetwork

# ── Parameter bounds (Table A) ────────────────────────────────────────────────

_BOUNDS_SIMPLE = {
    'beta0':  (0.00,  0.85),
    'beta1':  (-0.30, -0.10),
    'beta2':  (0.35,  0.95),
    'beta12': (0.05,  0.40),
    'lam1_0': (10.0,  65.0),
    'theta1': (0.0,   1.0),
    'lam2_0': (0.0,   50.0),
    'theta2': (0.0,   1.0),
}

# raw[5] encodes lam1_1 as a fraction of lam1_0: lam1_1 = lam1_0 * sigmoid(raw[5])
# raw[8] encodes lam2_1 as a fraction of lam2_0: lam2_1 = lam2_0 * sigmoid(raw[8])
# This guarantees the ordering constraints lam1_0 > lam1_1 and lam2_0 > lam2_1.


def _logit(v: float) -> float:
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return float(np.log(v / (1 - v)))


def _sig_enc(value: float, lo: float, hi: float) -> float:
    return _logit((value - lo) / (hi - lo))


def _sig_dec(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def params_to_raw(params: PDVParams) -> torch.Tensor:
    """Convert a PDVParams into a 10-element unconstrained tensor."""
    raw = [
        _sig_enc(params.beta0,  *_BOUNDS_SIMPLE['beta0']),
        _sig_enc(params.beta1,  *_BOUNDS_SIMPLE['beta1']),
        _sig_enc(params.beta2,  *_BOUNDS_SIMPLE['beta2']),
        _sig_enc(params.beta12, *_BOUNDS_SIMPLE['beta12']),
        _sig_enc(params.lam1_0, *_BOUNDS_SIMPLE['lam1_0']),
        _logit(params.lam1_1 / params.lam1_0),   # fraction of lam1_0
        _sig_enc(params.theta1, *_BOUNDS_SIMPLE['theta1']),
        _sig_enc(params.lam2_0, *_BOUNDS_SIMPLE['lam2_0']),
        _logit(params.lam2_1 / max(params.lam2_0, 1e-6)),  # fraction of lam2_0
        _sig_enc(params.theta2, *_BOUNDS_SIMPLE['theta2']),
    ]
    return torch.tensor(raw, dtype=torch.float32)


def raw_to_params_tensor(raw: torch.Tensor) -> torch.Tensor:
    """
    Decode a 10-element unconstrained tensor into a 10-element constrained
    parameter tensor (β₀, β₁, β₂, β₁₂, λ₁₀, λ₁₁, θ₁, λ₂₀, λ₂₁, θ₂).
    Fully differentiable — gradients flow through all sigmoid operations.
    """
    b0   = _sig_dec(raw[0], *_BOUNDS_SIMPLE['beta0'])
    b1   = _sig_dec(raw[1], *_BOUNDS_SIMPLE['beta1'])
    b2   = _sig_dec(raw[2], *_BOUNDS_SIMPLE['beta2'])
    b12  = _sig_dec(raw[3], *_BOUNDS_SIMPLE['beta12'])
    l1_0 = _sig_dec(raw[4], *_BOUNDS_SIMPLE['lam1_0'])
    l1_1 = l1_0 * torch.sigmoid(raw[5])           # in [0, lam1_0]
    th1  = _sig_dec(raw[6], *_BOUNDS_SIMPLE['theta1'])
    l2_0 = _sig_dec(raw[7], *_BOUNDS_SIMPLE['lam2_0'])
    l2_1 = l2_0 * torch.sigmoid(raw[8])           # in [0, lam2_0]
    th2  = _sig_dec(raw[9], *_BOUNDS_SIMPLE['theta2'])

    return torch.stack([b0, b1, b2, b12, l1_0, l1_1, th1, l2_0, l2_1, th2])


# ── Historical R factor initialisation ────────────────────────────────────────

def fetch_spx_history(n_years: int = 4) -> np.ndarray:
    """
    Download recent SPX daily closes and return simple returns,
    most recent first (index 0 = yesterday).
    """
    spx = yf.Ticker("^SPX")
    hist = spx.history(period=f"{n_years + 1}y")["Close"].dropna()
    returns = hist.pct_change().dropna().values  # simple returns
    return returns[::-1].copy()                  # most recent first


def compute_R_factors(
    lam1_0: torch.Tensor,
    lam1_1: torch.Tensor,
    lam2_0: torch.Tensor,
    lam2_1: torch.Tensor,
    returns: torch.Tensor,           # shape (T,), most recent first
    dt: float = 1 / 252,
) -> torch.Tensor:
    """
    Compute R factor initial values from historical returns.

        R_{n,j,0} = λ_{n,j} · Σ_{i=0}^{T-2} exp(-λ_{n,j} · i · dt) · r_{-i}^n

    Returns a 4-element tensor: (R1_0_0, R1_1_0, R2_0_0, R2_1_0).
    Differentiable with respect to λ values.
    """
    T = len(returns)
    lags = torch.arange(T, dtype=torch.float32) * dt

    def weighted(lam, power):
        w = lam * torch.exp(-lam * lags)
        return (w * returns ** power).sum()

    R1_0 = weighted(lam1_0, 1)
    R1_1 = weighted(lam1_1, 1)
    R2_0 = weighted(lam2_0, 2)
    R2_1 = weighted(lam2_1, 2)

    return torch.stack([R1_0, R1_1, R2_0, R2_1])


# ── Network input builders ────────────────────────────────────────────────────

def _build_inputs(theta14: torch.Tensor,
                  T_vals: torch.Tensor,
                  K_vals: torch.Tensor) -> torch.Tensor:
    """
    Build (N, 16) input tensor for a network from a 14-dim theta,
    a vector of maturities T_vals, and a vector of strikes/moneyness K_vals.
    All three must broadcast to the same length N.
    """
    N = len(T_vals)
    theta_rep = theta14.unsqueeze(0).expand(N, -1)   # (N, 14)
    return torch.cat([theta_rep, T_vals.unsqueeze(1), K_vals.unsqueeze(1)], dim=1)


# ── Joint loss function (equation 5) ─────────────────────────────────────────

def joint_loss(
    raw: torch.Tensor,               # 10 unconstrained params
    returns: torch.Tensor,           # historical SPX returns
    market: dict,                    # from fetcher.get_market_snapshot()
    spx_net: PDVNetwork,
    vix_net: PDVNetwork,
    w_spx: float = 1.0,
    w_vix_f: float = 1.0,
    w_vix_c: float = 1.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Compute the joint SPX/VIX calibration loss (equation 5 of the paper).

    SPX term  : mean squared relative error in implied vol
    VIX F term: mean squared relative error in futures price
    VIX C term: mean squared relative error in call price

    Note: VIX options use price-ratio loss rather than IV-ratio loss
    (avoiding the non-differentiable Black-Scholes inversion step).
    """
    # Decode constrained parameters
    theta_M = raw_to_params_tensor(raw)   # (10,)

    # Compute R factors from λ values and history
    R = compute_R_factors(
        lam1_0=theta_M[4], lam1_1=theta_M[5],
        lam2_0=theta_M[7], lam2_1=theta_M[8],
        returns=returns,
    )                                      # (4,)

    theta14 = torch.cat([theta_M, R])     # full 14-dim parameter vector

    spx_net.eval()
    vix_net.eval()

    total = torch.tensor(0.0, device=device)
    n_terms = 0

    # ── SPX smiles ────────────────────────────────────────────────────────────
    for smile in market.get('spx_smiles', []):
        if len(smile['strikes']) == 0:
            continue
        T_v = torch.full((len(smile['strikes']),), smile['T'], dtype=torch.float32)
        K_v = torch.tensor(smile['strikes'] / smile['forward'],  # normalise to [0,1]
                           dtype=torch.float32)
        iv_mkt = torch.tensor(smile['ivs'], dtype=torch.float32)

        inputs  = _build_inputs(theta14, T_v, K_v)
        iv_mdl  = spx_net(inputs).squeeze(1)

        total   = total + w_spx * ((iv_mdl / iv_mkt - 1) ** 2).mean()
        n_terms += 1

    # ── VIX futures ───────────────────────────────────────────────────────────
    for fut in market.get('vix_futures', []):
        T_v    = torch.tensor([fut['T']], dtype=torch.float32)
        m_v    = torch.tensor([1.0],      dtype=torch.float32)   # ATM moneyness
        F_mkt  = torch.tensor([fut['futures_price']], dtype=torch.float32)

        inputs     = _build_inputs(theta14, T_v, m_v)
        out        = vix_net(inputs)               # (1, 2)
        F_mdl      = out[:, 0]

        total   = total + w_vix_f * ((F_mdl / F_mkt - 1) ** 2).mean()
        n_terms += 1

    # ── VIX options ───────────────────────────────────────────────────────────
    for smile in market.get('vix_smiles', []):
        if len(smile['moneyness']) == 0:
            continue
        T_v   = torch.full((len(smile['moneyness']),), smile['T'], dtype=torch.float32)
        m_v   = torch.tensor(smile['moneyness'], dtype=torch.float32)
        # Use mid price as target (average of bid and ask)
        C_mkt = torch.tensor((smile['bid'] + smile['ask']) / 2, dtype=torch.float32)

        inputs = _build_inputs(theta14, T_v, m_v)
        out    = vix_net(inputs)           # (N, 2)
        C_mdl  = out[:, 1]

        # Filter out near-zero prices to avoid division instability
        mask   = C_mkt > 1e-4
        if mask.sum() > 0:
            total   = total + w_vix_c * ((C_mdl[mask] / C_mkt[mask] - 1) ** 2).mean()
            n_terms += 1

    return total / max(n_terms, 1)


# ── Main calibrator ───────────────────────────────────────────────────────────

def calibrate(
    spx_net: PDVNetwork,
    vix_net: PDVNetwork,
    market: dict,
    init_params: PDVParams | None = None,
    n_steps: int = 300,
    lr: float = 5e-3,
    w_spx: float = 1.0,
    w_vix_f: float = 1.0,
    w_vix_c: float = 0.5,
    device: str = 'cpu',
    verbose: bool = True,
    progress_callback=None,    # optional fn(step, loss) for dashboard updates
) -> tuple[PDVParams, list[float]]:
    """
    Run joint SPX/VIX calibration.

    Parameters
    ----------
    spx_net         : trained SPX implied vol network
    vix_net         : trained VIX futures/call network
    market          : snapshot from fetcher.get_market_snapshot()
    init_params     : starting PDVParams (defaults to paper's Oct 2009 values)
    n_steps         : gradient descent iterations
    lr              : Adam learning rate
    progress_callback : called each step with (step, loss_value)

    Returns
    -------
    (calibrated PDVParams, loss history)
    """
    from src.model.pdv_model import PAPER_PARAMS

    if init_params is None:
        init_params = PAPER_PARAMS

    # Historical returns for R factor computation
    if verbose:
        print("Fetching SPX history for R factor initialisation...")
    raw_returns = fetch_spx_history(n_years=4)
    returns = torch.tensor(raw_returns, dtype=torch.float32)

    # Initialise unconstrained parameters
    raw = params_to_raw(init_params).requires_grad_(True)
    optimiser = torch.optim.Adam([raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_steps, eta_min=lr * 0.01
    )

    loss_history = []

    for step in range(1, n_steps + 1):
        optimiser.zero_grad()
        loss = joint_loss(raw, returns, market, spx_net, vix_net,
                          w_spx, w_vix_f, w_vix_c, device)
        loss.backward()
        optimiser.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and step % 50 == 0:
            lr_now = optimiser.param_groups[0]['lr']
            print(f"  Step {step:4d} | loss {loss_val:.6f} | lr {lr_now:.2e}")

        if progress_callback is not None:
            progress_callback(step, loss_val)

    # Decode final parameters
    with torch.no_grad():
        theta_M = raw_to_params_tensor(raw)
        R = compute_R_factors(
            lam1_0=theta_M[4], lam1_1=theta_M[5],
            lam2_0=theta_M[7], lam2_1=theta_M[8],
            returns=returns,
        )

    v = theta_M.tolist()
    r = R.tolist()

    result = PDVParams(
        beta0=v[0],  beta1=v[1],  beta2=v[2],  beta12=v[3],
        lam1_0=v[4], lam1_1=v[5], theta1=v[6],
        lam2_0=v[7], lam2_1=v[8], theta2=v[9],
        R1_0_0=r[0], R1_1_0=r[1], R2_0_0=r[2], R2_1_0=r[3],
    )

    if verbose:
        print(f"\nCalibration complete. Final loss: {loss_history[-1]:.6f}")
        print(f"  β₀={result.beta0:.4f}  β₁={result.beta1:.4f}  "
              f"β₂={result.beta2:.4f}  β₁₂={result.beta12:.4f}")
        print(f"  λ₁₀={result.lam1_0:.2f}  λ₁₁={result.lam1_1:.2f}  "
              f"θ₁={result.theta1:.4f}")
        print(f"  λ₂₀={result.lam2_0:.2f}  λ₂₁={result.lam2_1:.2f}  "
              f"θ₂={result.theta2:.4f}")

    return result, loss_history
