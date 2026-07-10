import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class RegimeGate(nn.Module):
    def __init__(self, hidden_dim=16, n_modalities=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_modalities),
        )

    def forward(self, regime):
        logits = self.net(regime)
        return torch.softmax(logits, dim=-1)
