"""
Training loop for the SPX and VIX neural networks.

Uses:
  - Adam optimiser with an initial learning rate of 1e-3
  - ReduceLROnPlateau scheduler: halves lr when validation loss plateaus
  - MSE loss on normalised targets (so SPX and VIX losses are comparable)
  - Train/validation split of 80/20
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.training.networks import PDVNetwork


def _to_tensors(inputs: np.ndarray, targets: np.ndarray):
    return (torch.tensor(inputs,  dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))


def train(
    model: PDVNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    patience: int = 20,
    device: str = 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Train a PDVNetwork on the provided (inputs, targets) dataset.

    Parameters
    ----------
    model        : PDVNetwork (SPX or VIX)
    inputs       : (N, 16) float32 array
    targets      : (N, 1) or (N, 2) float32 array
    epochs       : maximum training epochs
    batch_size   : mini-batch size
    lr           : initial learning rate for Adam
    val_fraction : fraction of data held out for validation
    patience     : early stopping: halt if val loss doesn't improve for
                   this many epochs
    device       : 'cpu', 'cuda', or 'mps'

    Returns
    -------
    dict with 'train_loss' and 'val_loss' history lists
    """
    model = model.to(device)

    # Fit and store standardisation statistics
    model.fit_scaler(inputs, targets)

    # Normalise inputs and targets for training
    inp_mean = model.input_mean.cpu().numpy()
    inp_std  = model.input_std.cpu().numpy()
    tgt_mean = model.target_mean.cpu().numpy()
    tgt_std  = model.target_std.cpu().numpy()

    inputs_norm  = (inputs  - inp_mean) / inp_std
    targets_norm = (targets - tgt_mean) / tgt_std

    X, y = _to_tensors(inputs_norm, targets_norm)
    dataset = TensorDataset(X, y)

    n_val   = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 4)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=patience // 2, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    train_history, val_history = [], []
    best_val   = float('inf')
    best_state = None
    wait       = 0

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model.forward_normalised(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model.forward_normalised(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step(val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if verbose and epoch % 10 == 0:
            lr_now = optimiser.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} "
                  f"| val {val_loss:.6f} | lr {lr_now:.2e}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val loss {best_val:.6f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {'train_loss': train_history, 'val_loss': val_history}


def save_model(model: PDVNetwork, path: str) -> None:
    """Save model weights to a .pt file."""
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def load_model(model: PDVNetwork, path: str,
               device: str = 'cpu') -> PDVNetwork:
    """Load model weights from a .pt file."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate(
    model: PDVNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate a trained model and return MAE and RMSE in original units.
    """
    model.eval().to(device)
    X = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X).cpu().numpy()

    errors = np.abs(preds - targets)
    return {
        'mae':  errors.mean(axis=0),
        'rmse': np.sqrt((errors**2).mean(axis=0)),
        'preds': preds,
    }
