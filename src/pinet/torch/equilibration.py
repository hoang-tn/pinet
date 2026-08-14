"""Modified Ruiz equilibration (Torch)."""

import torch
from torch import Tensor

from .dataclasses import EquilibrationParams

EXPECTED_MATRIX_NDIM = 2


def ruiz_equilibration(
    a_mat: Tensor, params: EquilibrationParams
) -> tuple[Tensor, Tensor, Tensor]:
    """Perform modified Ruiz equilibration on a 2D matrix.

    Args:
        a_mat: Input matrix of shape ``(n_r, n_c)``.
        params: Equilibration parameters.

    Returns:
        A triple ``(scaled_a_mat, d_r, d_c)`` such that
        ``scaled_a_mat = diag(d_r) @ a_mat @ diag(d_c)``.
    """
    assert a_mat.ndim == EXPECTED_MATRIX_NDIM, (
        "Input matrix to equilibration must be 2-dimensional."
    )
    params.validate()

    scaled_a_mat = a_mat
    d_r = torch.ones(a_mat.shape[0], dtype=a_mat.dtype, device=a_mat.device)
    d_c = torch.ones(a_mat.shape[1], dtype=a_mat.dtype, device=a_mat.device)
    best_criterion = 1.0
    d_r_best = d_r
    d_c_best = d_c
    alpha = (
        (a_mat.shape[0] / a_mat.shape[1]) ** (1 / (2 * params.ord))
        if params.col_scaling
        else 1.0
    )

    ord_p: float | str = params.ord
    if params.ord == float("inf"):
        ord_p = "inf"

    for _ in range(params.max_iter):
        if params.update_mode == "Gauss":
            row_norms = torch.linalg.norm(scaled_a_mat, dim=1, ord=ord_p)
            row_factors = torch.where(row_norms > 0, torch.sqrt(row_norms), 1.0)
            d_r = d_r / row_factors
            scaled_a_mat = scaled_a_mat / row_factors[:, None]

            col_norms = torch.linalg.norm(scaled_a_mat, dim=0, ord=ord_p)
            col_factors = alpha * torch.where(col_norms > 0, torch.sqrt(col_norms), 1.0)
            d_c = d_c / col_factors
            scaled_a_mat = scaled_a_mat / col_factors[None, :]
        else:
            row_norms = torch.linalg.norm(scaled_a_mat, dim=1, ord=ord_p)
            row_factors = torch.where(row_norms > 0, torch.sqrt(row_norms), 1.0)
            col_norms = torch.linalg.norm(scaled_a_mat, dim=0, ord=ord_p)
            col_factors = alpha * torch.where(col_norms > 0, torch.sqrt(col_norms), 1.0)
            d_r = d_r / row_factors
            d_c = d_c / col_factors
            scaled_a_mat = scaled_a_mat / row_factors[:, None]
            scaled_a_mat = scaled_a_mat / col_factors[None, :]

        new_row_norms = torch.linalg.norm(scaled_a_mat, dim=1, ord=ord_p)
        new_col_norms = torch.linalg.norm(scaled_a_mat, dim=0, ord=ord_p)
        term_criterion = torch.maximum(
            1 - new_row_norms.min() / new_row_norms.max(),
            1 - new_col_norms.min() / new_col_norms.max(),
        )
        term_value = float(term_criterion.detach())
        if term_value < best_criterion:
            best_criterion = term_value
            d_r_best = d_r
            d_c_best = d_c
        if term_value < params.tol:
            break

    scaled_a_mat_best = a_mat * d_r_best[:, None]
    scaled_a_mat_best = scaled_a_mat_best * d_c_best[None, :]

    if params.safeguard:
        cond_a_mat = torch.linalg.cond(a_mat)
        cond_scaled = torch.linalg.cond(scaled_a_mat_best)
        if float(cond_scaled.detach()) > float(cond_a_mat.detach()):
            scaled_a_mat_best = a_mat
            d_r_best = torch.ones_like(d_r_best)
            d_c_best = torch.ones_like(d_c_best)

    return scaled_a_mat_best, d_r_best, d_c_best
