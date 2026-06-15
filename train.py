"""
Full training pipeline — generates data, trains both networks, saves .pt files.

Usage:
    python3.10 train.py [--quick]

    --quick   Tiny dataset for a smoke test (~2 min on a laptop).
              Omit for a production run (hours, ideally on a GPU machine).

Output files (in checkpoints/):
    spx_network.pt   — SPX implied vol network weights
    vix_network.pt   — VIX futures + call network weights
    spx_data.npz     — cached SPX training data (skip regeneration on reruns)
    vix_data.npz     — cached VIX training data

Workflow:
    1. Generate training data via Monte Carlo / LSMC (slow, done once).
    2. Train SPX network on (θ, T, K) → IV pairs.
    3. Train VIX network on (θ, T, m) → (F, C) pairs.
    4. Save .pt checkpoints — upload these to the Streamlit dashboard.
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.training.data_generation import generate_spx_data, generate_vix_data
from src.training.networks import build_spx_network, build_vix_network
from src.training.trainer import train, evaluate, save_model

# ── Configuration ─────────────────────────────────────────────────────────────

QUICK = dict(
    # ~2 minutes on a laptop, good for verifying the pipeline
    n_params_spx        = 50,
    n_params_vix        = 30,
    n_maturities        = 4,
    n_strikes_spx       = 6,
    n_strikes_vix       = 10,
    mc_paths            = 4_000,
    vix_n_out           = 1_024,
    vix_n_sub           = 64,
    vix_n_inner         = 128,
    epochs              = 50,
    batch_size          = 128,
)

PRODUCTION = dict(
    # Hours on a laptop, ~30-60 min on a GPU machine
    # Closer to what the paper uses (they generated millions of points on HPC)
    n_params_spx        = 2_000,
    n_params_vix        = 1_000,
    n_maturities        = 11,
    n_strikes_spx       = 8,
    n_strikes_vix       = 20,
    mc_paths            = 20_000,
    vix_n_out           = 4_096,
    vix_n_sub           = 256,
    vix_n_inner         = 512,
    epochs              = 300,
    batch_size          = 2_048,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_data(data: dict, path: str) -> None:
    np.savez(path, **{k: v for k, v in data.items()})
    print(f"  Cached to {path}")


def _load_data(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return 'mps'
    return 'cpu'


# ── Main ──────────────────────────────────────────────────────────────────────

def main(quick: bool = False):
    cfg = QUICK if quick else PRODUCTION
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = _detect_device()
    print(f"Device: {device}")
    print(f"Mode:   {'quick smoke test' if quick else 'production'}\n")

    spx_cache = os.path.join(CHECKPOINT_DIR, 'spx_data.npz')
    vix_cache = os.path.join(CHECKPOINT_DIR, 'vix_data.npz')

    # ── Step 1: Generate or load SPX data ────────────────────────────────────
    if os.path.exists(spx_cache):
        print("Loading cached SPX data...")
        spx_data = _load_data(spx_cache)
    else:
        print("Generating SPX training data (Monte Carlo)...")
        spx_data = generate_spx_data(
            n_params               = cfg['n_params_spx'],
            n_maturities_per_param = cfg['n_maturities'],
            n_strikes_per_maturity = cfg['n_strikes_spx'],
            mc_paths               = cfg['mc_paths'],
            seed                   = 0,
        )
        _save_data(spx_data, spx_cache)

    print(f"SPX dataset: {spx_data['inputs'].shape[0]:,} points  "
          f"(IV range [{spx_data['targets'].min():.3f}, "
          f"{spx_data['targets'].max():.3f}])\n")

    # ── Step 2: Generate or load VIX data ────────────────────────────────────
    if os.path.exists(vix_cache):
        print("Loading cached VIX data...")
        vix_data = _load_data(vix_cache)
    else:
        print("Generating VIX training data (LSMC)...")
        vix_data = generate_vix_data(
            n_params               = cfg['n_params_vix'],
            n_strikes_per_maturity = cfg['n_strikes_vix'],
            n_out                  = cfg['vix_n_out'],
            n_sub                  = cfg['vix_n_sub'],
            n_inner                = cfg['vix_n_inner'],
            seed                   = 1,
        )
        _save_data(vix_data, vix_cache)

    print(f"VIX dataset: {vix_data['inputs'].shape[0]:,} points  "
          f"(futures range [{vix_data['targets'][:,0].min():.3f}, "
          f"{vix_data['targets'][:,0].max():.3f}])\n")

    # ── Step 3: Train SPX network ─────────────────────────────────────────────
    print("=" * 50)
    print("Training SPX network...")
    print("=" * 50)
    spx_net = build_spx_network()
    spx_hist = train(
        spx_net,
        spx_data['inputs'],
        spx_data['targets'],
        epochs     = cfg['epochs'],
        batch_size = cfg['batch_size'],
        device     = device,
        verbose    = True,
    )
    spx_path = os.path.join(CHECKPOINT_DIR, 'spx_network.pt')
    save_model(spx_net, spx_path)

    spx_eval = evaluate(spx_net, spx_data['inputs'], spx_data['targets'], device=device)
    print(f"SPX MAE (in-sample): {spx_eval['mae'][0]:.5f} vol points\n")

    # ── Step 4: Train VIX network ─────────────────────────────────────────────
    print("=" * 50)
    print("Training VIX network...")
    print("=" * 50)
    vix_net = build_vix_network()
    vix_hist = train(
        vix_net,
        vix_data['inputs'],
        vix_data['targets'],
        epochs     = cfg['epochs'],
        batch_size = cfg['batch_size'],
        device     = device,
        verbose    = True,
    )
    vix_path = os.path.join(CHECKPOINT_DIR, 'vix_network.pt')
    save_model(vix_net, vix_path)

    vix_eval = evaluate(vix_net, vix_data['inputs'], vix_data['targets'], device=device)
    print(f"VIX futures MAE (in-sample): {vix_eval['mae'][0]:.5f}")
    print(f"VIX call    MAE (in-sample): {vix_eval['mae'][1]:.5f}\n")

    print("=" * 50)
    print("Done. Upload these files to the Streamlit dashboard:")
    print(f"  {spx_path}")
    print(f"  {vix_path}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run a fast smoke test with tiny data')
    args = parser.parse_args()
    main(quick=args.quick)
