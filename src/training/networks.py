"""
Neural network architectures for the SPX and VIX pricers.

Both networks follow Figure 1 of the paper:
  - Input layer: 16-dimensional  (14 model params + maturity T + strike/moneyness)
  - 5 hidden layers of 256 units each
  - ELU activations (smooth and well-behaved for financial functions)
  - Output: 1-dimensional for SPX (implied vol), 2-dimensional for VIX (F, C)

Inputs are standardised (zero mean, unit variance) before entering the network.
Standardisation statistics are learned from the training set and stored with
the model so they can be applied consistently at inference time.
"""

import torch
import torch.nn as nn
import numpy as np


class PDVNetwork(nn.Module):
    """
    Generic feedforward network for the PDV model pricing problem.

    Configurable for either the SPX or VIX task by changing n_outputs.
    """

    def __init__(
        self,
        n_inputs: int = 16,
        n_outputs: int = 1,
        hidden_size: int = 256,
        n_layers: int = 5,
    ):
        super().__init__()

        layers = []
        in_dim = n_inputs
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ELU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, n_outputs))

        self.net = nn.Sequential(*layers)

        # Standardisation buffers (set via fit_scaler, used in forward)
        self.register_buffer('input_mean', torch.zeros(n_inputs))
        self.register_buffer('input_std',  torch.ones(n_inputs))
        self.register_buffer('target_mean', torch.zeros(n_outputs))
        self.register_buffer('target_std',  torch.ones(n_outputs))

    def fit_scaler(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Compute and store standardisation statistics from training data."""
        self.input_mean.copy_(torch.tensor(inputs.mean(0),  dtype=torch.float32))
        self.input_std.copy_( torch.tensor(inputs.std(0).clip(1e-8), dtype=torch.float32))
        self.target_mean.copy_(torch.tensor(targets.mean(0), dtype=torch.float32))
        self.target_std.copy_( torch.tensor(targets.std(0).clip(1e-8), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on raw (unstandardised) inputs.
        Returns predictions in the original (unstandardised) target units.
        """
        x_norm = (x - self.input_mean) / self.input_std
        y_norm = self.net(x_norm)
        return y_norm * self.target_std + self.target_mean

    def forward_normalised(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass when inputs are already standardised (used during training)."""
        return self.net(x_norm)


def build_spx_network(hidden_size: int = 256, n_layers: int = 5) -> PDVNetwork:
    """SPX network: (θ, T, K) → implied vol."""
    return PDVNetwork(n_inputs=16, n_outputs=1,
                      hidden_size=hidden_size, n_layers=n_layers)


def build_vix_network(hidden_size: int = 256, n_layers: int = 5) -> PDVNetwork:
    """VIX network: (θ, T, m) → (futures price, call price)."""
    return PDVNetwork(n_inputs=16, n_outputs=2,
                      hidden_size=hidden_size, n_layers=n_layers)
