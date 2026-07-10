import numpy as np
import torch


def _kernel_matrix(kernel_module, z_a, z_b):
    with torch.no_grad():
        return kernel_module(z_a, z_b).evaluate().detach().cpu().numpy()


def exact_variance_decomposition(model, x_train, x_test, noise_var):
    zf_tr, zt_tr, zs_tr = model.encode(x_train)
    zf_te, zt_te, zs_te = model.encode(x_test)

    kernels = {
        "fundamental": (model.kernel_f, zf_tr, zf_te),
        "technical": (model.kernel_t, zt_tr, zt_te),
        "sentiment": (model.kernel_s, zs_tr, zs_te),
    }

    K_full = np.zeros((x_train.shape[0], x_train.shape[0]))
    k_star_full = {}
    k_train_per_modality = {}
    k_diag_star = {}

    for name, (kernel_module, z_tr, z_te) in kernels.items():
        K_mm = _kernel_matrix(kernel_module, z_tr, z_tr)
        K_full += K_mm
        k_train_per_modality[name] = K_mm
        k_star_full[name] = _kernel_matrix(kernel_module, z_te, z_tr)
        k_diag_star[name] = _kernel_matrix(kernel_module, z_te, z_te).diagonal()

    K_full += noise_var * np.eye(x_train.shape[0])
    K_inv = np.linalg.inv(K_full)

    n_test = x_test.shape[0]
    modality_names = list(kernels.keys())
    diagonal_terms = {name: np.zeros(n_test) for name in modality_names}
    total_variance = np.zeros(n_test)
    prior_variance = np.zeros(n_test)

    k_star_sum = sum(k_star_full[name] for name in modality_names)
    prior_diag_sum = sum(k_diag_star[name] for name in modality_names)

    reduction_full = np.einsum("ij,jk,ik->i", k_star_sum, K_inv, k_star_sum)
    total_variance = prior_diag_sum - reduction_full
    prior_variance = prior_diag_sum

    for name in modality_names:
        reduction_own = np.einsum(
            "ij,jk,ik->i", k_star_full[name], K_inv, k_star_full[name]
        )
        diagonal_terms[name] = k_diag_star[name] - reduction_own

    interaction = total_variance - sum(diagonal_terms[name] for name in modality_names)

    return {
        "total_variance": total_variance,
        "prior_variance": prior_variance,
        "diagonal": diagonal_terms,
        "interaction": interaction,
        "modality_names": modality_names,
    }


def dominant_modality(diagonal_terms):
    names = list(diagonal_terms.keys())
    stacked = np.stack([diagonal_terms[n] for n in names], axis=1)
    idx = np.argmax(stacked, axis=1)
    return np.array([names[i] for i in idx])
