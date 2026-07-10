import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from arch import arch_model
import gpytorch
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel


def build_lstm_baseline(input_dim, seq_len, hidden_units=32):
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, input_dim)),
            layers.LSTM(hidden_units, return_sequences=False),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def make_sequences(features, target, seq_len):
    X, y = [], []
    values = features.values
    tgt = target.values
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len : i])
        y.append(tgt[i])
    return np.array(X), np.array(y)


def empirical_residual_quantile_intervals(residuals, level=0.9):
    lower_q = (1 - level) / 2
    upper_q = 1 - lower_q
    lower = np.quantile(residuals, lower_q)
    upper = np.quantile(residuals, upper_q)
    return lower, upper


def fit_garch_volatility(returns_pct):
    am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="StudentsT")
    res = am.fit(disp="off")
    return res


class SingleKernelGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_single_kernel_gp(train_x, train_y, n_iter=200, lr=0.05):
    likelihood = GaussianLikelihood()
    model = SingleKernelGP(train_x, train_y, likelihood)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    return model, likelihood
