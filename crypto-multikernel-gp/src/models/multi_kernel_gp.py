import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import StudentTLikelihood

from .encoders import ModalityEncoder, RegimeGate


class ModalitySlices:
    def __init__(self, fundamental_dim, technical_dim, sentiment_dim):
        self.f = slice(0, fundamental_dim)
        self.t = slice(fundamental_dim, fundamental_dim + technical_dim)
        self.s = slice(
            fundamental_dim + technical_dim,
            fundamental_dim + technical_dim + sentiment_dim,
        )
        self.regime_idx = fundamental_dim + technical_dim + sentiment_dim


class MultiKernelAdditiveGP(ApproximateGP):
    def __init__(self, inducing_points, slices, latent_dim=8, use_regime_gate=True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.slices = slices
        self.use_regime_gate = use_regime_gate

        self.encoder_f = ModalityEncoder(slices.f.stop - slices.f.start, latent_dim=latent_dim)
        self.encoder_t = ModalityEncoder(slices.t.stop - slices.t.start, latent_dim=latent_dim)
        self.encoder_s = ModalityEncoder(slices.s.stop - slices.s.start, latent_dim=latent_dim)

        self.mean_module = ConstantMean()
        self.kernel_f = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=latent_dim))
        self.kernel_t = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=latent_dim))
        self.kernel_s = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=latent_dim))

        if use_regime_gate:
            self.gate = RegimeGate(n_modalities=3)

    def encode(self, x):
        xf, xt, xs = x[..., self.slices.f], x[..., self.slices.t], x[..., self.slices.s]
        zf, zt, zs = self.encoder_f(xf), self.encoder_t(xt), self.encoder_s(xs)
        return zf, zt, zs

    def per_modality_kernels(self, x):
        zf, zt, zs = self.encode(x)
        return {"fundamental": (self.kernel_f, zf), "technical": (self.kernel_t, zt), "sentiment": (self.kernel_s, zs)}

    def forward(self, x):
        zf, zt, zs = self.encode(x)
        kff = self.kernel_f(zf)
        ktt = self.kernel_t(zt)
        kss = self.kernel_s(zs)

        if self.use_regime_gate:
            regime = x[..., self.slices.regime_idx : self.slices.regime_idx + 1]
            weights = self.gate(regime)
            w_mean = weights.mean(dim=0)
            covar = kff.mul(w_mean[0] * 3) + ktt.mul(w_mean[1] * 3) + kss.mul(w_mean[2] * 3)
        else:
            covar = kff + ktt + kss

        mean = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def build_likelihood(deg_free=4.0):
    from gpytorch.constraints import GreaterThan
    likelihood = StudentTLikelihood(noise_constraint=GreaterThan(1e-2))
    likelihood._set_deg_free(torch.tensor(float(deg_free)))
    likelihood.raw_deg_free.requires_grad_(False)
    return likelihood
