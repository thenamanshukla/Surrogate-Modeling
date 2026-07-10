import unittest
import numpy as np


def make_random_psd_kernel_matrix(n, d, seed, lengthscale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    sq_dists = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-sq_dists / (2 * lengthscale ** 2))
    return x, K


def kernel_matrix_between(x_a, x_b, lengthscale=1.0):
    sq_dists = ((x_a[:, None, :] - x_b[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-sq_dists / (2 * lengthscale ** 2))


def exact_decomposition_numpy(kernel_matrices_train, kernel_matrices_star_train, kernel_diag_star, noise_var):
    modality_names = list(kernel_matrices_train.keys())
    n_train = next(iter(kernel_matrices_train.values())).shape[0]
    n_test = next(iter(kernel_matrices_star_train.values())).shape[0]

    K_full = sum(kernel_matrices_train[name] for name in modality_names) + noise_var * np.eye(n_train)
    K_inv = np.linalg.inv(K_full)

    k_star_sum = sum(kernel_matrices_star_train[name] for name in modality_names)
    prior_diag_sum = sum(kernel_diag_star[name] for name in modality_names)

    total_variance = prior_diag_sum - np.einsum("ij,jk,ik->i", k_star_sum, K_inv, k_star_sum)

    diagonal = {}
    for name in modality_names:
        k_star = kernel_matrices_star_train[name]
        reduction_own = np.einsum("ij,jk,ik->i", k_star, K_inv, k_star)
        diagonal[name] = kernel_diag_star[name] - reduction_own

    interaction = total_variance - sum(diagonal[name] for name in modality_names)

    return {"total_variance": total_variance, "diagonal": diagonal, "interaction": interaction}


class TestExactDecompositionIdentity(unittest.TestCase):
    def setUp(self):
        n_train, n_test, d = 40, 15, 3
        self.modality_names = ["fundamental", "technical", "sentiment"]
        self.kernel_train = {}
        self.kernel_star_train = {}
        self.kernel_diag_star = {}

        for i, name in enumerate(self.modality_names):
            x_train, K_train = make_random_psd_kernel_matrix(n_train, d, seed=100 + i, lengthscale=0.7 + 0.3 * i)
            rng = np.random.default_rng(200 + i)
            x_test = rng.normal(size=(n_test, d))
            outputscale = 0.5 + i * 0.3
            self.kernel_train[name] = outputscale * K_train
            self.kernel_star_train[name] = outputscale * kernel_matrix_between(x_test, x_train, lengthscale=0.7 + 0.3 * i)
            diag_dists = np.zeros(n_test)
            self.kernel_diag_star[name] = outputscale * np.exp(-diag_dists)

        self.noise_var = 0.05
        self.decomp = exact_decomposition_numpy(
            self.kernel_train, self.kernel_star_train, self.kernel_diag_star, self.noise_var
        )

    def test_diagonal_plus_interaction_equals_total_to_machine_precision(self):
        reconstructed = sum(self.decomp["diagonal"][name] for name in self.modality_names) + self.decomp["interaction"]
        np.testing.assert_allclose(reconstructed, self.decomp["total_variance"], rtol=1e-9, atol=1e-12)

    def test_total_variance_is_non_negative(self):
        self.assertTrue((self.decomp["total_variance"] >= -1e-8).all())

    def test_total_variance_matches_direct_full_kernel_computation(self):
        K_full_direct = sum(self.kernel_train.values()) + self.noise_var * np.eye(40)
        K_inv_direct = np.linalg.inv(K_full_direct)
        k_star_direct = sum(self.kernel_star_train.values())
        prior_diag_direct = sum(self.kernel_diag_star.values())
        expected_total = prior_diag_direct - np.einsum("ij,jk,ik->i", k_star_direct, K_inv_direct, k_star_direct)
        np.testing.assert_allclose(self.decomp["total_variance"], expected_total, rtol=1e-9)

    def test_single_dominant_modality_gets_majority_share_when_others_are_flat(self):
        n_train, n_test, d = 40, 10, 2
        rng = np.random.default_rng(5)
        x_train_dom, K_dom = make_random_psd_kernel_matrix(n_train, d, seed=5, lengthscale=0.5)
        x_test_dom = rng.normal(size=(n_test, d))

        kernel_train = {
            "fundamental": 0.01 * np.ones((n_train, n_train)) + 0.001 * np.eye(n_train),
            "technical": 0.01 * np.ones((n_train, n_train)) + 0.001 * np.eye(n_train),
            "sentiment": 3.0 * K_dom,
        }
        kernel_star_train = {
            "fundamental": 0.01 * np.ones((n_test, n_train)),
            "technical": 0.01 * np.ones((n_test, n_train)),
            "sentiment": 3.0 * kernel_matrix_between(x_test_dom, x_train_dom, lengthscale=0.5),
        }
        kernel_diag_star = {
            "fundamental": 0.01 * np.ones(n_test),
            "technical": 0.01 * np.ones(n_test),
            "sentiment": 3.0 * np.ones(n_test),
        }
        decomp = exact_decomposition_numpy(kernel_train, kernel_star_train, kernel_diag_star, noise_var=0.05)
        avg_diag = {name: decomp["diagonal"][name].mean() for name in kernel_train}
        self.assertGreater(avg_diag["sentiment"], avg_diag["fundamental"])
        self.assertGreater(avg_diag["sentiment"], avg_diag["technical"])

    def test_interaction_can_be_negative_when_modalities_are_redundant(self):
        n_train, n_test, d = 30, 10, 2
        x_train, K_shared = make_random_psd_kernel_matrix(n_train, d, seed=11, lengthscale=0.6)
        rng = np.random.default_rng(12)
        x_test = rng.normal(size=(n_test, d))
        K_star_shared = kernel_matrix_between(x_test, x_train, lengthscale=0.6)
        diag_shared = np.ones(n_test)

        kernel_train = {"fundamental": K_shared, "technical": K_shared.copy(), "sentiment": 0.001 * np.eye(n_train)}
        kernel_star_train = {
            "fundamental": K_star_shared,
            "technical": K_star_shared.copy(),
            "sentiment": np.zeros((n_test, n_train)),
        }
        kernel_diag_star = {"fundamental": diag_shared, "technical": diag_shared.copy(), "sentiment": np.full(n_test, 0.001)}
        decomp = exact_decomposition_numpy(kernel_train, kernel_star_train, kernel_diag_star, noise_var=0.01)
        self.assertLess(decomp["interaction"].mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
