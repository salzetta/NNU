"""
Guyon-Lekeufack four-factor Markov path-dependent volatility (PDV) model.

State variables:
    S       : asset price
    R1_0    : fast component of the trend factor R1
    R1_1    : slow component of the trend factor R1
    R2_0    : fast component of the variability factor R2
    R2_1    : slow component of the variability factor R2

Aggregated factors:
    R1 = (1 - theta1) * R1_0 + theta1 * R1_1
    R2 = (1 - theta2) * R2_0 + theta2 * R2_1

Volatility function:
    sigma(R1, R2) = beta0 + beta1*R1 + beta2*sqrt(R2) + beta12*R1^2 * 1_{R1>0}

SDEs (Euler-Maruyama discretization):
    dS       = S * sigma * dW
    dR1_j    = lam1_j * (sigma * dW  -  R1_j * dt)
    dR2_j    = lam2_j * (sigma^2     -  R2_j) * dt
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PDVParams:
    """
    Parameters for the four-factor PDV model.
    Bounds follow Table A of the paper.
    """
    # Volatility function coefficients
    beta0:  float   # baseline volatility,         in [0, 0.85]
    beta1:  float   # trend coefficient,            in [-0.30, -0.10]
    beta2:  float   # variability coefficient,      in [0.35, 0.95]
    beta12: float   # parabolic (right-tail) term,  in [0.05, 0.40]

    # Decay rates and mixing weight for R1
    lam1_0: float   # fast decay rate,  in [10, 65]
    lam1_1: float   # slow decay rate,  in [0, 35]  (must be < lam1_0)
    theta1: float   # mixing weight,    in [0, 1]

    # Decay rates and mixing weight for R2
    lam2_0: float   # fast decay rate,  in [0, 50]
    lam2_1: float   # slow decay rate,  in [0, 15]  (must be < lam2_0)
    theta2: float   # mixing weight,    in [0, 1]

    # Initial factor values (computed from historical price data at calibration time)
    R1_0_0: float   # initial fast R1,  in [-1.62, 0.88]
    R1_1_0: float   # initial slow R1,  in [-1.05, 0.71]
    R2_0_0: float   # initial fast R2,  in [0, 0.11]
    R2_1_0: float   # initial slow R2,  in [0, 0.11]

    def __post_init__(self):
        if self.lam1_0 <= self.lam1_1:
            raise ValueError("lam1_0 must be greater than lam1_1 (fast > slow)")
        if self.lam2_0 <= self.lam2_1:
            raise ValueError("lam2_0 must be greater than lam2_1 (fast > slow)")


# Calibrated parameters from Table C of the paper (October 21, 2009)
PAPER_PARAMS = PDVParams(
    beta0=0.0850,  beta1=-0.2637, beta2=0.6614, beta12=0.2164,
    lam1_0=50.05,  lam1_1=7.52,   theta1=0.8888,
    lam2_0=5.70,   lam2_1=0.29,   theta2=0.9750,
    R1_0_0=-0.0163, R1_1_0=-0.4337,
    R2_0_0=0.0398,  R2_1_0=0.0581,
)


_SIGMA_MAX = 5.0   # hard cap: prevents overflow in σ² and exp(σ·dW)


def compute_sigma(R1: np.ndarray, R2: np.ndarray, p: PDVParams) -> np.ndarray:
    """
    Instantaneous volatility: sigma(R1, R2).
    Clamped to [1e-6, _SIGMA_MAX] for numerical stability.
    """
    # Clip R1 before squaring to prevent overflow in the beta12 term
    R1_clip = np.clip(R1, -50.0, 50.0)
    vol = (p.beta0
           + p.beta1 * R1_clip
           + p.beta2 * np.sqrt(np.maximum(R2, 0.0))
           + p.beta12 * R1_clip**2 * (R1_clip > 0))
    return np.clip(vol, 1e-6, _SIGMA_MAX)


def simulate(
    params: PDVParams,
    n_paths: int,
    n_steps: int,
    dt: float = 1 / 252,
    S0: float = 1.0,
    seed: Optional[int] = None,
    store_paths: bool = True,
) -> dict:
    """
    Simulate the PDV model via Euler-Maruyama.

    Parameters
    ----------
    params      : PDVParams
    n_paths     : number of Monte Carlo paths
    n_steps     : number of time steps
    dt          : step size in years (default 1/252 = one trading day)
    S0          : initial asset price (default 1.0, i.e. normalised)
    seed        : random seed for reproducibility
    store_paths : if True, store full paths; if False, only keep terminal state
                  (more memory efficient for large training runs)

    Returns
    -------
    dict with keys:
        'S'     : asset price paths,     shape (n_paths, n_steps+1) or (n_paths,)
        'R1'    : aggregated R1 paths,   same shape
        'R2'    : aggregated R2 paths,   same shape
        'vol'   : instantaneous vol,     same shape
        'R1_0'  : fast R1 factor,        terminal only if store_paths=False
        'R1_1'  : slow R1 factor,        terminal only
        'R2_0'  : fast R2 factor,        terminal only
        'R2_1'  : slow R2 factor,        terminal only
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    p = params

    # Initialise state vectors (one value per path)
    S   = np.full(n_paths, float(S0))
    R1_0 = np.full(n_paths, p.R1_0_0)
    R1_1 = np.full(n_paths, p.R1_1_0)
    R2_0 = np.full(n_paths, p.R2_0_0)
    R2_1 = np.full(n_paths, p.R2_1_0)

    # Aggregate factors and initial volatility
    R1  = (1 - p.theta1) * R1_0 + p.theta1 * R1_1
    R2  = (1 - p.theta2) * R2_0 + p.theta2 * R2_1
    sig = compute_sigma(R1, R2, p)

    if store_paths:
        T = n_steps + 1
        S_arr   = np.empty((n_paths, T))
        R1_arr  = np.empty((n_paths, T))
        R2_arr  = np.empty((n_paths, T))
        vol_arr = np.empty((n_paths, T))

        S_arr[:, 0]   = S
        R1_arr[:, 0]  = R1
        R2_arr[:, 0]  = R2
        vol_arr[:, 0] = sig

    for i in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt

        # Asset price: log-Euler scheme avoids S going negative
        S = S * np.exp(sig * dW - 0.5 * sig**2 * dt)

        # R1 components: stochastic (driven by same dW as the asset)
        R1_0 = R1_0 + p.lam1_0 * (sig * dW - R1_0 * dt)
        R1_1 = R1_1 + p.lam1_1 * (sig * dW - R1_1 * dt)

        # R2 components: deterministic ODE (mean-reversion towards sigma^2)
        R2_0 = R2_0 + p.lam2_0 * (sig**2 - R2_0) * dt
        R2_1 = R2_1 + p.lam2_1 * (sig**2 - R2_1) * dt

        # Aggregate and update volatility for next step
        R1  = (1 - p.theta1) * R1_0 + p.theta1 * R1_1
        R2  = (1 - p.theta2) * R2_0 + p.theta2 * R2_1
        sig = compute_sigma(R1, R2, p)

        if store_paths:
            S_arr[:, i+1]   = S
            R1_arr[:, i+1]  = R1
            R2_arr[:, i+1]  = R2
            vol_arr[:, i+1] = sig

    if store_paths:
        return {
            'S': S_arr, 'R1': R1_arr, 'R2': R2_arr, 'vol': vol_arr,
            'R1_0': R1_0, 'R1_1': R1_1, 'R2_0': R2_0, 'R2_1': R2_1,
        }
    else:
        return {
            'S': S, 'R1': R1, 'R2': R2, 'vol': sig,
            'R1_0': R1_0, 'R1_1': R1_1, 'R2_0': R2_0, 'R2_1': R2_1,
        }


def compute_initial_factors(
    returns: np.ndarray,
    lam1_0: float, lam1_1: float,
    lam2_0: float, lam2_1: float,
    dt: float = 1 / 252,
) -> dict:
    """
    Compute initial factor values R_{n,j,0} from historical daily returns.

    The paper uses 4 years (T=1008 trading days) of history:
        R_{1,j,0} = lam_{1,j} * sum_{i=0}^{T-2} exp(-lam_{1,j} * i/252) * r_{-i}
        R_{2,j,0} = lam_{2,j} * sum_{i=0}^{T-2} exp(-lam_{2,j} * i/252) * r_{-i}^2

    Parameters
    ----------
    returns : array of daily simple returns (r_t = S_t/S_{t-1} - 1),
              ordered from most recent [0] to oldest [-1]
    """
    T = len(returns)
    lags = np.arange(T) * dt

    def weighted_sum(lam, power):
        weights = lam * np.exp(-lam * lags)
        return np.sum(weights * returns**power)

    return {
        'R1_0_0': weighted_sum(lam1_0, 1),
        'R1_1_0': weighted_sum(lam1_1, 1),
        'R2_0_0': weighted_sum(lam2_0, 2),
        'R2_1_0': weighted_sum(lam2_1, 2),
    }
