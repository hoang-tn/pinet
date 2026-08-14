"""Differentiable batched QP layer with a fast dense PDIPM.

``QPFunction`` is a drop-in for ``qpth.qp.QPFunction``: it solves

.. math::

    \\hat z = \\arg\\min_z \\tfrac{1}{2} z^T Q z + p^T z
    \\quad \\text{s.t.} \\quad G z \\le h,\\; A z = b

and backpropagates through the KKT system at the solution (OptNet / qpth).
Pass unbatched ``Q``, ``G``, and ``A`` when they are shared across the
batch -- that is the common projection-layer case and avoids qpth's
per-sample refactor of identical blocks.
"""

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from pinet.torch.solver.pdipm import (
    factor_schur,
    make_workspace,
    outer,
    pdipm_forward,
    precompute_schur,
    solve_kkt,
)


class QPFunction:
    """Differentiable batched QP solver (qpth-compatible).

    Call as ``QPFunction(max_iter=20)(q_mat, p, g_mat, h, a_mat, b)``.

    Attributes:
        eps: Interior-point residual tolerance.
        max_iter: Maximum Mehrotra iterations.
        check_q_spd: If ``True``, Cholesky-check ``Q`` before solving.
    """

    eps: float
    max_iter: int
    check_q_spd: bool

    def __init__(
        self,
        eps: float = 1e-12,
        max_iter: int = 20,
        check_q_spd: bool = False,
    ) -> None:
        """Store solver options.

        Args:
            eps: Interior-point residual tolerance.
            max_iter: Maximum Mehrotra iterations.
            check_q_spd: If ``True``, Cholesky-check ``Q`` before solving.
        """
        self.eps = eps
        self.max_iter = max_iter
        self.check_q_spd = check_q_spd

    def __call__(
        self,
        q_mat: Tensor,
        p: Tensor,
        g_mat: Tensor,
        h: Tensor,
        a_mat: Tensor,
        b: Tensor,
    ) -> Tensor:
        """Solve the QP.

        Args:
            q_mat: Quadratic term.
            p: Linear term.
            g_mat: Inequality matrix.
            h: Inequality right-hand side.
            a_mat: Equality matrix.
            b: Equality right-hand side.

        Returns:
            Primal solution ``zhat`` of shape ``(B, n)``.
        """
        return _qp_autograd(
            q_mat,
            p,
            g_mat,
            h,
            a_mat,
            b,
            self.eps,
            self.max_iter,
            self.check_q_spd,
        )


def _qp_autograd(
    q_mat: Tensor,
    p: Tensor,
    g_mat: Tensor,
    h: Tensor,
    a_mat: Tensor,
    b: Tensor,
    eps: float,
    max_iter: int,
    check_q_spd: bool,
) -> Tensor:
    """Apply the custom autograd QP primitive.

    Args:
        q_mat: Quadratic term.
        p: Linear term.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        eps: Interior-point residual tolerance.
        max_iter: Maximum Mehrotra iterations.
        check_q_spd: If ``True``, Cholesky-check ``Q`` before solving.

    Returns:
        Primal solution.
    """

    class _QPAutograd(Function):
        @staticmethod
        def forward(
            ctx: Any,
            q_in: Tensor,
            p_in: Tensor,
            g_in: Tensor,
            h_in: Tensor,
            a_in: Tensor,
            b_in: Tensor,
        ) -> Tensor:
            """Solve the QP and stash KKT factors for the backward pass.

            Args:
                ctx: Autograd context.
                q_in: Quadratic term.
                p_in: Linear term.
                g_in: Inequality matrix.
                h_in: Inequality right-hand side.
                a_in: Equality matrix.
                b_in: Equality right-hand side.

            Returns:
                Primal solution ``zhat`` of shape ``(B, n)``.
            """
            if check_q_spd:
                torch.linalg.cholesky(q_in)
            ctx.q_batched = q_in.ndim == 3
            ctx.p_batched = p_in.ndim == 2
            ctx.g_batched = g_in.ndim == 3
            ctx.h_batched = h_in.ndim == 2
            ctx.a_batched = a_in.ndim == 3
            ctx.b_batched = b_in.ndim == 2
            zhat, nu, lam, slack = pdipm_forward(
                q_in, p_in, g_in, h_in, a_in, b_in, eps=eps, max_iter=max_iter
            )
            ctx.save_for_backward(
                zhat, q_in, p_in, g_in, h_in, a_in, b_in, nu, lam, slack
            )
            return zhat

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            """Differentiate the QP through the KKT system at the solution.

            Args:
                ctx: Autograd context from the forward pass.
                *grad_outputs: Incoming cotangents of ``zhat``.

            Returns:
                Cotangents of ``(q_mat, p, g_mat, h, a_mat, b)``.
            """
            grad_zhat = grad_outputs[0]
            zhat, q_in, _, g_in, _, a_in, b_in, nu, lam, slack = ctx.saved_tensors
            cache = precompute_schur(q_in, g_in, a_in)
            n_batch = zhat.shape[0]
            workspace = make_workspace(cache, n_batch)
            d_scale = lam.clamp(min=1e-8) / slack.clamp(min=1e-8)
            factor = factor_schur(workspace, slack.clamp(min=1e-8) / lam.clamp(min=1e-8))
            zeros_ineq = zhat.new_zeros(n_batch, cache.nineq)
            zeros_eq = zhat.new_zeros(n_batch, cache.neq)
            dx, _, dlam, dnu = solve_kkt(
                workspace,
                factor,
                d_scale,
                grad_zhat,
                zeros_ineq,
                zeros_ineq,
                zeros_eq,
            )
            d_q = 0.5 * (outer(dx, zhat) + outer(zhat, dx))
            d_p = dx
            d_g = outer(dlam, zhat) + outer(lam, dx)
            d_h = -dlam
            d_a = outer(dnu, zhat) + outer(nu, dx)
            d_b = -dnu
            if not ctx.q_batched:
                d_q = d_q.mean(dim=0)
            if not ctx.p_batched:
                d_p = d_p.mean(dim=0)
            if not ctx.g_batched:
                d_g = d_g.mean(dim=0)
            if not ctx.h_batched:
                d_h = d_h.mean(dim=0)
            if a_in.numel() == 0:
                d_a_out: Tensor | None = None
            elif ctx.a_batched:
                d_a_out = d_a
            else:
                d_a_out = d_a.mean(dim=0)
            if b_in.numel() == 0:
                d_b_out: Tensor | None = None
            elif ctx.b_batched:
                d_b_out = d_b
            else:
                d_b_out = d_b.mean(dim=0)
            return d_q, d_p, d_g, d_h, d_a_out, d_b_out

    return _QPAutograd.apply(q_mat, p, g_mat, h, a_mat, b)


def solve_qp(
    q_mat: Tensor,
    p: Tensor,
    g_mat: Tensor,
    h: Tensor,
    a_mat: Tensor,
    b: Tensor,
    *,
    eps: float = 1e-12,
    max_iter: int = 20,
    check_q_spd: bool = False,
) -> Tensor:
    """Solve a batch of dense QPs.

    Args:
        q_mat: Quadratic term, ``(n, n)`` or ``(B, n, n)``.
        p: Linear term, ``(n,)`` or ``(B, n)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        h: Inequality right-hand side, ``(nineq,)`` or ``(B, nineq)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``.
        b: Equality right-hand side, ``(neq,)`` or ``(B, neq)``.
        eps: Interior-point residual tolerance.
        max_iter: Maximum Mehrotra iterations.
        check_q_spd: If ``True``, Cholesky-check ``Q`` before solving.

    Returns:
        Primal solution of shape ``(B, n)``.
    """
    return QPFunction(eps=eps, max_iter=max_iter, check_q_spd=check_q_spd)(
        q_mat, p, g_mat, h, a_mat, b
    )


def project_affine(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    *,
    eps: float = 1e-12,
    max_iter: int = 20,
) -> Tensor:
    """Project rows of ``x`` onto ``A y = b``, ``G y <= h``.

    Args:
        x: Points to project, ``(B, n)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``.
        b: Equality right-hand side, ``(neq,)`` or ``(B, neq)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        h: Inequality right-hand side, ``(nineq,)`` or ``(B, nineq)``.
        eps: Interior-point residual tolerance.
        max_iter: Maximum Mehrotra iterations.

    Returns:
        Projected points of shape ``(B, n)``.
    """
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return solve_qp(eye, -x, g_mat, h, a_mat, b, eps=eps, max_iter=max_iter)
