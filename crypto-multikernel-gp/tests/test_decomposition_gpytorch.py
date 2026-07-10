import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import gpytorch
    from src.models.multi_kernel_gp import MultiKernelAdditiveGP, ModalitySlices, build_likelihood
    from src.models.decomposition import exact_variance_decomposition, dominant_modality
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch/gpytorch not installed in this environment")
class TestExactDecompositionAgainstRealModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.fund_dim, self.tech_dim, self.sent_dim = 6, 9, 4
        self.slices = ModalitySlices(self.fund_dim, self.tech_dim, self.sent_dim)
        total_dim = self.fund_dim + self.tech_dim + self.sent_dim + 1
        self.n_train = 40
        self.n_test = 10
        self.train_x = torch.randn(self.n_train, total_dim)
        self.test_x = torch.randn(self.n_test, total_dim)

        inducing = self.train_x[:20].clone()
        self.model = MultiKernelAdditiveGP(inducing, self.slices, use_regime_gate=False)
        self.model.eval()
        self.noise_var = 0.05

    def test_decomposition_sums_to_total_to_machine_precision(self):
        decomp = exact_variance_decomposition(self.model, self.train_x, self.test_x, self.noise_var)
        reconstructed = sum(decomp["diagonal"][name] for name in decomp["modality_names"]) + decomp["interaction"]
        np.testing.assert_allclose(reconstructed, decomp["total_variance"], rtol=1e-6, atol=1e-8)

    def test_decomposition_total_is_non_negative(self):
        decomp = exact_variance_decomposition(self.model, self.train_x, self.test_x, self.noise_var)
        self.assertTrue((decomp["total_variance"] >= -1e-6).all())

    def test_dominant_modality_labels_are_valid(self):
        decomp = exact_variance_decomposition(self.model, self.train_x, self.test_x, self.noise_var)
        dominant = dominant_modality(decomp["diagonal"])
        self.assertEqual(len(dominant), self.n_test)
        self.assertTrue(set(dominant).issubset(set(decomp["modality_names"])))

    def test_decomposition_matches_manual_full_kernel_computation(self):
        with torch.no_grad():
            zf_tr, zt_tr, zs_tr = self.model.encode(self.train_x)
            zf_te, zt_te, zs_te = self.model.encode(self.test_x)

            Kff = self.model.kernel_f(zf_tr, zf_tr).evaluate().numpy()
            Ktt = self.model.kernel_t(zt_tr, zt_tr).evaluate().numpy()
            Kss = self.model.kernel_s(zs_tr, zs_tr).evaluate().numpy()
            K_full = Kff + Ktt + Kss + self.noise_var * np.eye(self.n_train)
            K_inv = np.linalg.inv(K_full)

            kf_star = self.model.kernel_f(zf_te, zf_tr).evaluate().numpy()
            kt_star = self.model.kernel_t(zt_te, zt_tr).evaluate().numpy()
            ks_star = self.model.kernel_s(zs_te, zs_tr).evaluate().numpy()
            k_star_sum = kf_star + kt_star + ks_star

            diag_f = self.model.kernel_f(zf_te, zf_te).evaluate().numpy().diagonal()
            diag_t = self.model.kernel_t(zt_te, zt_te).evaluate().numpy().diagonal()
            diag_s = self.model.kernel_s(zs_te, zs_te).evaluate().numpy().diagonal()
            prior_diag_sum = diag_f + diag_t + diag_s

            expected_total = prior_diag_sum - np.einsum("ij,jk,ik->i", k_star_sum, K_inv, k_star_sum)

        decomp = exact_variance_decomposition(self.model, self.train_x, self.test_x, self.noise_var)
        np.testing.assert_allclose(decomp["total_variance"], expected_total, rtol=1e-5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch/gpytorch not installed in this environment")
class TestModelForwardPassShapes(unittest.TestCase):
    def test_forward_pass_produces_valid_multivariate_normal(self):
        slices = ModalitySlices(6, 9, 4)
        total_dim = 6 + 9 + 4 + 1
        train_x = torch.randn(30, total_dim)
        inducing = train_x[:15].clone()
        model = MultiKernelAdditiveGP(inducing, slices)
        likelihood = build_likelihood()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(train_x)
            self.assertEqual(output.mean.shape[0], 30)
            posterior = likelihood(output)
            self.assertTrue(hasattr(posterior, "mean"))
            self.assertTrue(hasattr(posterior, "variance"))


if __name__ == "__main__":
    unittest.main()
