"""Pure tensor kernels for projection, used by constraints and the ADMM loop."""

import torch
from torch import Tensor


def equality_project(x: Tensor, a_mat: Tensor, a_pinv: Tensor, b: Tensor) -> Tensor:
    """Orthogonal projection onto ``a_mat @ x == b``.

    Args:
        x: Point to project, shape ``(B, n, 1)``.
        a_mat: Equality matrix, shape ``(#B, m, n)``.
        a_pinv: Pseudoinverse of ``a_mat``, shape ``(#B, n, m)``.
        b: Equality right-hand side, shape ``(#B, m, 1)``.

    Returns:
        Projected point of shape ``(B, n, 1)``.
    """
    return x - a_pinv @ (a_mat @ x - b)


def equality_cv(x: Tensor, a_mat: Tensor, b: Tensor) -> Tensor:
    """Infinity-norm equality residual.

    Args:
        x: Point to evaluate, shape ``(B, n, 1)``.
        a_mat: Equality matrix, shape ``(#B, m, n)``.
        b: Equality right-hand side, shape ``(#B, m, 1)``.

    Returns:
        Constraint violation of shape ``(B, 1, 1)``.
    """
    return (a_mat @ x - b).abs().amax(dim=1, keepdim=True)


def box_project(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Clip ``x`` into ``[lb, ub]`` coordinate-wise.

    Args:
        x: Point to project, shape ``(B, n, 1)``.
        lb: Lower bounds (use ``-inf`` on free coordinates).
        ub: Upper bounds (use ``inf`` on free coordinates).

    Returns:
        Clipped point of shape ``(B, n, 1)``.
    """
    return torch.minimum(torch.maximum(x, lb), ub)


def box_cv(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Maximum box violation.

    Args:
        x: Point to evaluate, shape ``(B, n, 1)``.
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Constraint violation of shape ``(B, 1, 1)``.
    """
    violation = torch.maximum(x - ub, lb - x)
    return torch.clamp(violation, min=0).amax(dim=1, keepdim=True)


def soc_project(
    x: Tensor,
    mask_u: Tensor,
    mask_t: Tensor,
    a_full: Tensor,
    b_full: Tensor,
    eps: Tensor,
) -> Tensor:
    """Project onto ``||x[mask_u] + a||_2 <= x[mask_t] + b``.

    Args:
        x: Point to project, shape ``(B, n, 1)``.
        mask_u: Boolean mask for the cone vector, shape ``(1, n, 1)``.
        mask_t: Boolean mask for the cone scalar, shape ``(1, n, 1)``.
        a_full: Offset ``a`` scattered onto ``mask_u``.
        b_full: Offset ``b`` scattered onto ``mask_t``.
        eps: Positive stabilizer for the direction ``u / ||u||``.

    Returns:
        Projected point of shape ``(B, n, 1)``.
    """
    zeros = torch.zeros_like(x)
    u = torch.where(mask_u, x + a_full, zeros)
    t = torch.where(mask_t, x + b_full, zeros).sum(dim=1, keepdim=True)
    norm_u = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    direction_u = u / (norm_u + eps)
    half = (t + norm_u) / 2
    proj3_u = half * direction_u
    proj3_t = half
    when1 = norm_u <= t
    when2 = norm_u <= -t
    proj_u = torch.where(when1, u, torch.where(when2, zeros, proj3_u))
    proj_t = torch.where(when1, t, torch.where(when2, torch.zeros_like(t), proj3_t))
    x_new = torch.where(mask_u, proj_u - a_full, x)
    return torch.where(mask_t, proj_t - b_full, x_new)


def soc_cv(
    x: Tensor,
    mask_u: Tensor,
    mask_t: Tensor,
    a_full: Tensor,
    b_full: Tensor,
) -> Tensor:
    """SOC violation ``max(0, ||u+a||_2 - (t+b))``.

    Args:
        x: Point to evaluate, shape ``(B, n, 1)``.
        mask_u: Boolean mask for the cone vector, shape ``(1, n, 1)``.
        mask_t: Boolean mask for the cone scalar, shape ``(1, n, 1)``.
        a_full: Offset ``a`` scattered onto ``mask_u``.
        b_full: Offset ``b`` scattered onto ``mask_t``.

    Returns:
        Constraint violation of shape ``(B, 1, 1)``.
    """
    zeros = torch.zeros_like(x)
    u = torch.where(mask_u, x + a_full, zeros)
    t = torch.where(mask_t, x + b_full, zeros).sum(dim=1, keepdim=True)
    norm_u = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    return torch.clamp(norm_u - t, min=0)


def ineq_cv(x: Tensor, c_mat: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Maximum affine-inequality violation.

    Args:
        x: Point to evaluate, shape ``(B, n, 1)``.
        c_mat: Inequality matrix, shape ``(#B, n_ineq, n)``.
        lb: Lower bounds, shape ``(#B, n_ineq, 1)``.
        ub: Upper bounds, shape ``(#B, n_ineq, 1)``.

    Returns:
        Constraint violation of shape ``(B, 1, 1)``.
    """
    cx = c_mat @ x
    violation = torch.maximum(cx - ub, lb - cx)
    return torch.clamp(violation, min=0).amax(dim=1, keepdim=True)
