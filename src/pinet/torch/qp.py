"""Differentiable batched QP layer with a fast dense PDIPM.

``QPFunction`` is a drop-in for ``qpth.qp.QPFunction``: it solves

.. math::

    \\hat z = \\arg\\min_z \\tfrac{1}{2} z^T Q z + p^T z
    \\quad \\text{s.t.} \\quad G z \\le h,\\; A z = b

and backpropagates through the KKT system at the solution (OptNet / qpth).
Pass unbatched ``Q``, ``G``, and ``A`` when they are shared across the
batch -- that is the common projection-layer case and avoids qpth's
per-sample refactor of identical blocks.

``project_affine`` defaults to a hybrid ADMM plus active-set polish that
targets qpth-level feasibility at near-ADMM cost. Use ``solver="pdipm"``
or ``QPFunction`` when the quadratic term is not the identity.
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
from pinet.torch.solver.polish import (
    HYBRID_DEFAULT_MAX_REPAIR,
    HYBRID_DEFAULT_N_ADMM,
    HYBRID_DEFAULT_SLACK_TOL,
    hybrid_forward,
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
            return _kkt_cotangents(
                grad_zhat,
                zhat,
                q_in,
                g_in,
                a_in,
                b_in,
                nu,
                lam,
                slack,
                ctx.q_batched,
                ctx.p_batched,
                ctx.g_batched,
                ctx.h_batched,
                ctx.a_batched,
                ctx.b_batched,
            )

    return _QPAutograd.apply(q_mat, p, g_mat, h, a_mat, b)


def _kkt_cotangents(
    grad_zhat: Tensor,
    zhat: Tensor,
    q_mat: Tensor,
    g_mat: Tensor,
    a_mat: Tensor,
    b: Tensor,
    nu: Tensor,
    lam: Tensor,
    slack: Tensor,
    q_batched: bool,
    p_batched: bool,
    g_batched: bool,
    h_batched: bool,
    a_batched: bool,
    b_batched: bool,
) -> tuple[Tensor | None, ...]:
    """Differentiate a QP through the KKT system at ``zhat``.

    Args:
        grad_zhat: Incoming cotangent of the primal solution.
        zhat: Primal solution.
        q_mat: Quadratic term.
        g_mat: Inequality matrix.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        nu: Equality dual.
        lam: Inequality dual.
        slack: Inequality slack.
        q_batched: Whether ``q_mat`` was batched.
        p_batched: Whether ``p`` was batched.
        g_batched: Whether ``g_mat`` was batched.
        h_batched: Whether ``h`` was batched.
        a_batched: Whether ``a_mat`` was batched.
        b_batched: Whether ``b`` was batched.

    Returns:
        Cotangents of ``(q_mat, p, g_mat, h, a_mat, b)``.
    """
    cache = precompute_schur(q_mat, g_mat, a_mat)
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
    if not q_batched:
        d_q = d_q.mean(dim=0)
    if not p_batched:
        d_p = d_p.mean(dim=0)
    if not g_batched:
        d_g = d_g.mean(dim=0)
    if not h_batched:
        d_h = d_h.mean(dim=0)
    if a_mat.numel() == 0:
        d_a_out: Tensor | None = None
    elif a_batched:
        d_a_out = d_a
    else:
        d_a_out = d_a.mean(dim=0)
    if b.numel() == 0:
        d_b_out: Tensor | None = None
    elif b_batched:
        d_b_out = d_b
    else:
        d_b_out = d_b.mean(dim=0)
    return d_q, d_p, d_g, d_h, d_a_out, d_b_out


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
    solver: str = "hybrid",
    eps: float = 1e-12,
    max_iter: int = 20,
    n_admm: int = HYBRID_DEFAULT_N_ADMM,
    slack_tol: float = HYBRID_DEFAULT_SLACK_TOL,
    max_repair: int = HYBRID_DEFAULT_MAX_REPAIR,
) -> Tensor:
    """Project rows of ``x`` onto ``A y = b``, ``G y <= h``.

    ``solver="hybrid"`` (default) runs a cheap ADMM warm start, polishes
    the active set with a few identity-``Q`` KKT solves, and falls back to
    the PDIPM on rows that do not stabilize. That combination is the
    projection-layer path that aims for ADMM-like wall time at qpth
    accuracy. ``solver="pdipm"`` is the full interior-point solve, also
    used by ``QPFunction`` for general quadratic objectives.

    Args:
        x: Points to project, ``(B, n)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``.
        b: Equality right-hand side, ``(neq,)`` or ``(B, neq)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        h: Inequality right-hand side, ``(nineq,)`` or ``(B, nineq)``.
        solver: ``"hybrid"`` or ``"pdipm"``.
        eps: Interior-point residual tolerance (PDIPM and hybrid fallback).
        max_iter: Maximum Mehrotra iterations (PDIPM and hybrid fallback).
        n_admm: Douglas-Rachford iterations used by the hybrid warm start.
        slack_tol: Active-set slack threshold after ADMM.
        max_repair: Maximum hybrid add/drop sweeps.

    Returns:
        Projected points of shape ``(B, n)``.

    Raises:
        ValueError: If ``solver`` is not ``hybrid`` or ``pdipm``.
    """
    if solver == "pdipm":
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return solve_qp(eye, -x, g_mat, h, a_mat, b, eps=eps, max_iter=max_iter)
    if solver != "hybrid":
        raise ValueError(f"Unknown solver {solver!r}. Use 'hybrid' or 'pdipm'.")
    return _project_hybrid_autograd(
        x, a_mat, b, g_mat, h, eps, max_iter, n_admm, slack_tol, max_repair
    )


def _project_hybrid_autograd(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    eps: float,
    max_iter: int,
    n_admm: int,
    slack_tol: float,
    max_repair: int,
) -> Tensor:
    """Apply the hybrid projector with a KKT backward pass.

    Args:
        x: Points to project.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        eps: PDIPM residual tolerance used on rows that do not polish.
        max_iter: PDIPM iterations used on rows that do not polish.
        n_admm: Douglas-Rachford iterations used as a warm start.
        slack_tol: Active-set slack threshold after ADMM.
        max_repair: Maximum active-set add/drop sweeps.

    Returns:
        Projected points.
    """

    class _HybridAutograd(Function):
        @staticmethod
        def forward(
            ctx: Any,
            x_in: Tensor,
            g_in: Tensor,
            h_in: Tensor,
            a_in: Tensor,
            b_in: Tensor,
        ) -> Tensor:
            """Project with ADMM plus active-set polish.

            Args:
                ctx: Autograd context.
                x_in: Points to project.
                g_in: Inequality matrix.
                h_in: Inequality right-hand side.
                a_in: Equality matrix.
                b_in: Equality right-hand side.

            Returns:
                Projected points.
            """
            ctx.x_batched = x_in.ndim == 2
            ctx.g_batched = g_in.ndim == 3
            ctx.h_batched = h_in.ndim == 2
            ctx.a_batched = a_in.ndim == 3
            ctx.b_batched = b_in.ndim == 2
            zhat, nu, lam, slack = hybrid_forward(
                x_in,
                a_in,
                b_in,
                g_in,
                h_in,
                n_admm=n_admm,
                slack_tol=slack_tol,
                max_repair=max_repair,
                eps=eps,
                max_iter=max_iter,
            )
            eye = torch.eye(x_in.shape[-1], dtype=x_in.dtype, device=x_in.device)
            ctx.save_for_backward(zhat, eye, g_in, a_in, b_in, nu, lam, slack)
            return zhat

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            """Differentiate the projection through the KKT system.

            Args:
                ctx: Autograd context from the forward pass.
                *grad_outputs: Incoming cotangents of the projection.

            Returns:
                Cotangents of ``(x, g_mat, h, a_mat, b)``.
            """
            grad_zhat = grad_outputs[0]
            zhat, q_in, g_in, a_in, b_in, nu, lam, slack = ctx.saved_tensors
            d_q, d_p, d_g, d_h, d_a, d_b = _kkt_cotangents(
                grad_zhat,
                zhat,
                q_in,
                g_in,
                a_in,
                b_in,
                nu,
                lam,
                slack,
                False,
                ctx.x_batched,
                ctx.g_batched,
                ctx.h_batched,
                ctx.a_batched,
                ctx.b_batched,
            )
            del d_q
            d_x = None if d_p is None else -d_p
            return d_x, d_g, d_h, d_a, d_b

    return _HybridAutograd.apply(x, g_mat, h, a_mat, b)
