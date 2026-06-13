"""
Phase 1 sanity check: simulate the PDV model and plot key outputs.
Run from the NNU/ directory: python notebooks/01_model_exploration.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from src.model.pdv_model import PAPER_PARAMS, simulate

# ── Simulation ────────────────────────────────────────────────────────────────

N_PATHS = 200
N_STEPS = 252          # one trading year
paths = simulate(PAPER_PARAMS, n_paths=N_PATHS, n_steps=N_STEPS, seed=42)

time = np.linspace(0, 1, N_STEPS + 1)   # in years

# ── Plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Guyon-Lekeufack PDV model — 200 simulated paths (1 year)", fontsize=13)

# Asset price
ax = axes[0]
ax.plot(time, paths['S'].T, color='steelblue', alpha=0.15, linewidth=0.7)
ax.plot(time, np.median(paths['S'], axis=0), color='navy', linewidth=1.5, label='median')
ax.set_ylabel("Asset price S")
ax.legend()

# Instantaneous volatility (annualised)
ax = axes[1]
ax.plot(time, paths['vol'].T, color='tomato', alpha=0.15, linewidth=0.7)
ax.plot(time, np.median(paths['vol'], axis=0), color='darkred', linewidth=1.5, label='median')
ax.set_ylabel("Instantaneous vol σ")
ax.legend()

# R1 (trend) and R2 (variability) aggregated factors
ax = axes[2]
ax.plot(time, np.median(paths['R1'], axis=0), color='darkorange',
        linewidth=1.5, label='median R1 (trend)')
ax.plot(time, np.median(paths['R2'], axis=0), color='purple',
        linewidth=1.5, label='median R2 (variability)')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_ylabel("Factor value")
ax.set_xlabel("Time (years)")
ax.legend()

plt.tight_layout()
plt.savefig("notebooks/model_paths.png", dpi=150)
plt.show()
print("Saved to notebooks/model_paths.png")

# ── Quick stats ───────────────────────────────────────────────────────────────

final_vol = paths['vol'][:, -1]
print(f"\nFinal instantaneous vol — mean: {final_vol.mean():.4f}, "
      f"std: {final_vol.std():.4f}, "
      f"min: {final_vol.min():.4f}, "
      f"max: {final_vol.max():.4f}")

log_returns = np.diff(np.log(paths['S']), axis=1).flatten()
ann_vol = log_returns.std() * np.sqrt(252)
print(f"Realised annualised vol from log-returns: {ann_vol:.4f}")
