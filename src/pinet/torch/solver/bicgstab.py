"""Batched matrix-free BiCGSTAB matching JAX's whole-tensor inner products."""

from collections.abc import Callable

import torch
from torch import Tensor


def bicgstab(
    operator: Callable[[Tensor], Tensor],
    rhs: Tensor,
    maxiter: int,
    tol: float = 1e-12,
) -> Tensor:
    """Solve ``operator(x) = rhs`` with BiCGSTAB.

    Inner products are taken over the entire tensor (including the batch axis),
    matching ``jax.scipy.sparse.linalg.bicgstab``.

    Args:
        operator: Linear map with the same input and output shape as ``rhs``.
        rhs: Right-hand side.
        maxiter: Maximum number of iterations.
        tol: Absolute residual tolerance.

    Returns:
        Approximate solution with the same shape as ``rhs``.
    """
    x = torch.zeros_like(rhs)
    r = rhs - operator(x)
    r_hat = r
    rho = _dot(r_hat, r)
    p = r
    eps = rhs.new_tensor(1e-30)

    for _ in range(maxiter):
        residual = torch.linalg.vector_norm(r)
        if float(residual.detach()) < tol:
            break
        v = operator(p)
        rhat_v = _dot(r_hat, v)
        alpha = rho / torch.where(rhat_v.abs() > eps, rhat_v, eps)
        s = r - alpha * v
        t = operator(s)
        t_t = _dot(t, t)
        omega = _dot(t, s) / torch.where(t_t.abs() > eps, t_t, eps)
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_new = _dot(r_hat, r)
        beta = (rho_new / torch.where(rho.abs() > eps, rho, eps)) * (
            alpha / torch.where(omega.abs() > eps, omega, eps)
        )
        p = r + beta * (p - omega * v)
        rho = rho_new
    return x


def _dot(left: Tensor, right: Tensor) -> Tensor:
    """Full-tensor inner product as a 0-dim tensor.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        Scalar inner product.
    """
    return torch.sum(left * right)
