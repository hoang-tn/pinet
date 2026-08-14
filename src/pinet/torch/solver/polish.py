"""ADMM warm-start plus active-set polish for Euclidean projection.

A full primal-dual interior-point method matches qpth accuracy, but each
iteration refactors an ``(neq + nineq)`` Schur complement. Douglas-Rachford
(ADMM) is much cheaper per step and usually lands on the correct active
face; a few masked KKT solves then recover a qpth-quality point. Rows that
do not stabilize fall back to the PDIPM.
"""

import torch
from torch import Tensor

from pinet.constants import Constants
from pinet.torch.solver.pdipm import (
    as_batch_vec,
    mv,
    mv_t,
    pdipm_forward,
)

HYBRID_DEFAULT_N_ADMM = 50
HYBRID_DEFAULT_SLACK_TOL = 5e-3
HYBRID_DEFAULT_MAX_REPAIR = 8
HYBRID_RIDGE = 1e-14
HYBRID_ADD_TOL = 1e-10
HYBRID_DROP_TOL = 1e-10
HYBRID_CV_TOL = 1e-8


def _select_batch(tensor: Tensor, idx: Tensor, item_ndim: int) -> Tensor:
    """Index a coefficient along the batch axis when it is batched.

    Args:
        tensor: Unbatched or batched coefficient.
        idx: Batch indices to keep.
        item_ndim: Rank of one unbatched item.

    Returns:
        The selected batch, or ``tensor`` unchanged when it is shared.
    """
    if tensor.ndim == item_ndim + 1 and tensor.shape[0] > 1:
        return tensor.index_select(0, idx)
    return tensor


def _equality_project(x: Tensor, a_mat: Tensor, b: Tensor) -> Tensor:
    """Project rows of ``x`` onto ``A y = b``.

    Args:
        x: Points, ``(B, n)``.
        a_mat: Equality matrix.
        b: Equality right-hand side.

    Returns:
        Equality-feasible points.
    """
    if a_mat.numel() == 0:
        return x
    resid = mv(a_mat, x) - as_batch_vec(b, x.shape[0], int(a_mat.shape[-2]), x)
    aat = torch.matmul(a_mat, a_mat.transpose(-1, -2))
    nu = torch.linalg.solve(aat, resid.unsqueeze(-1)).squeeze(-1)
    return x - mv_t(a_mat, nu)


def _lifted_equality(a_mat: Tensor, g_mat: Tensor) -> tuple[Tensor, int, int]:
    """Build the lifted equality ``A y = b``, ``G y + s = h``.

    Args:
        a_mat: Equality matrix, possibly empty.
        g_mat: Inequality matrix.

    Returns:
        Tuple ``(a_lift, neq, nineq)``.
    """
    nineq = int(g_mat.shape[-2])
    neq = int(a_mat.shape[-2]) if a_mat.numel() else 0
    eye_s = torch.eye(nineq, dtype=g_mat.dtype, device=g_mat.device)
    if g_mat.ndim == 3:
        eye_s = eye_s.expand(g_mat.shape[0], nineq, nineq)
    if neq == 0:
        return torch.cat([g_mat, eye_s], dim=-1), 0, nineq
    zeros_eq = a_mat.new_zeros(*a_mat.shape[:-1], nineq)
    top = torch.cat([a_mat, zeros_eq], dim=-1)
    bottom = torch.cat([g_mat, eye_s], dim=-1)
    return torch.cat([top, bottom], dim=-2), neq, nineq


def _affine_admm(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    n_iter: int,
    sigma: float,
    omega: float,
) -> Tensor:
    """Douglas-Rachford on the lifted polytope ``A y = b``, ``G y <= h``.

    Args:
        x: Points to project, ``(B, n)``.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        n_iter: Number of ADMM iterations. ``0`` is the equality projection.
        sigma: ADMM stepsize.
        omega: Relaxation.

    Returns:
        Approximate projection of shape ``(B, n)``.
    """
    if n_iter <= 0:
        return _equality_project(x, a_mat, b)
    batch, dim = x.shape
    a_lift, neq, nineq = _lifted_equality(a_mat, g_mat)
    b_eq = as_batch_vec(b, batch, neq, x)
    b_ineq = as_batch_vec(h, batch, nineq, x)
    b_lift = torch.cat([b_eq, b_ineq], dim=-1)
    a_pinv = torch.linalg.pinv(a_lift)
    sk = x.new_zeros(batch, dim + nineq)
    two_sigma = 2.0 * sigma
    denom = 1.0 + two_sigma
    b_col = b_lift.unsqueeze(-1)
    for _ in range(n_iter):
        zk = sk.unsqueeze(-1) - a_pinv @ (a_lift @ sk.unsqueeze(-1) - b_col)
        zk = zk.squeeze(-1)
        reflect = 2.0 * zk - sk
        primal = (two_sigma * x + reflect[:, :dim]) / denom
        slack = reflect[:, dim:].clamp(min=0.0)
        tk = torch.cat([primal, slack], dim=-1)
        sk = sk + omega * (tk - zk)
    zk = sk.unsqueeze(-1) - a_pinv @ (a_lift @ sk.unsqueeze(-1) - b_col)
    return zk.squeeze(-1)[:, :dim]


def _constraint_products(a_mat: Tensor, g_mat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Cache ``A A^T``, ``G A^T``, and ``G G^T``.

    Args:
        a_mat: Equality matrix, possibly empty.
        g_mat: Inequality matrix.

    Returns:
        Tuple ``(aat, gat, ggt)``.
    """
    ggt = torch.matmul(g_mat, g_mat.transpose(-1, -2))
    if a_mat.numel() == 0:
        empty = g_mat.new_zeros(*g_mat.shape[:-2], 0, 0)
        gat = g_mat.new_zeros(*g_mat.shape[:-1], 0)
        return empty, gat, ggt
    aat = torch.matmul(a_mat, a_mat.transpose(-1, -2))
    gat = torch.matmul(g_mat, a_mat.transpose(-1, -2))
    return aat, gat, ggt


def _solve_face(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    active: Tensor,
    aat: Tensor,
    gat: Tensor,
    ggt: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve the identity-``Q`` KKT system on a masked active set.

    Args:
        x: Unprojected points.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        active: Boolean mask of active inequalities, ``(B, nineq)``.
        aat: ``A A^T``.
        gat: ``G A^T``.
        ggt: ``G G^T``.

    Returns:
        Tuple ``(y, lam, nu)``.
    """
    batch = x.shape[0]
    nineq = int(g_mat.shape[-2])
    neq = int(a_mat.shape[-2]) if a_mat.numel() else 0
    mask = active.to(dtype=x.dtype)
    kkt = neq + nineq
    s_mat = ggt.new_zeros(batch, kkt, kkt)
    rhs = x.new_zeros(batch, kkt)
    if neq > 0:
        s_mat[:, :neq, :neq] = aat
        s_eq_in = gat.transpose(-1, -2) * mask.unsqueeze(-2)
        s_mat[:, :neq, neq:] = s_eq_in
        s_mat[:, neq:, :neq] = s_eq_in.transpose(-1, -2)
        rhs[:, :neq] = mv(a_mat, x) - as_batch_vec(b, batch, neq, x)
    s_mat[:, neq:, neq:] = mask.unsqueeze(-1) * ggt * mask.unsqueeze(-2)
    s_mat[:, neq:, neq:].diagonal(dim1=-2, dim2=-1).add_(1.0 - mask)
    s_mat.diagonal(dim1=-2, dim2=-1).add_(HYBRID_RIDGE)
    rhs[:, neq:] = mask * (mv(g_mat, x) - as_batch_vec(h, batch, nineq, x))
    lu_data, pivots = torch.linalg.lu_factor(s_mat)
    dual = torch.linalg.lu_solve(lu_data, pivots, rhs.unsqueeze(-1)).squeeze(-1)
    nu = dual[:, :neq] if neq > 0 else x.new_zeros(batch, 0)
    lam = dual[:, neq:]
    y = x - mv_t(g_mat, lam)
    if neq > 0:
        y = y - mv_t(a_mat, nu)
    return y, lam, nu


def _active_set_polish(
    x: Tensor,
    y0: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    slack_tol: float,
    max_repair: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Repair the active set until the face KKT system is stable.

    Args:
        x: Unprojected points.
        y0: Warm start, typically from ADMM.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        slack_tol: Mark an inequality active when its slack is below this.
        max_repair: Maximum add/drop sweeps.

    Returns:
        Tuple ``(y, lam, nu, failed)``. ``failed`` is true for rows that
        did not stabilize or that violate primal/dual feasibility.
    """
    batch = x.shape[0]
    nineq = int(g_mat.shape[-2])
    neq = int(a_mat.shape[-2]) if a_mat.numel() else 0
    h_b = as_batch_vec(h, batch, nineq, x)
    active = (h_b - mv(g_mat, y0)) <= slack_tol
    aat, gat, ggt = _constraint_products(a_mat, g_mat)
    y = y0.clone()
    lam = x.new_zeros(batch, nineq)
    nu = x.new_zeros(batch, neq)
    dirty = torch.ones(batch, dtype=torch.bool, device=x.device)
    for _ in range(max_repair):
        if not bool(dirty.any()):
            break
        if bool(dirty.all()):
            y_s, lam_s, nu_s = _solve_face(x, a_mat, b, g_mat, h, active, aat, gat, ggt)
            y, lam, nu = y_s, lam_s, nu_s
            idx: Tensor | None = None
            g_s, h_s = g_mat, h
            active_s = active
        else:
            idx = dirty.nonzero(as_tuple=True)[0]
            x_s = x.index_select(0, idx)
            active_s = active.index_select(0, idx)
            g_s = _select_batch(g_mat, idx, 2)
            h_s = _select_batch(h, idx, 1)
            a_s = _select_batch(a_mat, idx, 2)
            b_s = _select_batch(b, idx, 1)
            aat_s = _select_batch(aat, idx, 2)
            gat_s = _select_batch(gat, idx, 2)
            ggt_s = _select_batch(ggt, idx, 2)
            y_s, lam_s, nu_s = _solve_face(
                x_s, a_s, b_s, g_s, h_s, active_s, aat_s, gat_s, ggt_s
            )
            y.index_copy_(0, idx, y_s)
            lam.index_copy_(0, idx, lam_s)
            nu.index_copy_(0, idx, nu_s)
        viol = mv(g_s, y_s) - as_batch_vec(h_s, y_s.shape[0], nineq, x)
        drop = active_s & (lam_s < -HYBRID_DROP_TOL)
        add = (~active_s) & (viol > HYBRID_ADD_TOL)
        add = add & ~drop.any(dim=-1, keepdim=True)
        changed = add.any(dim=-1) | drop.any(dim=-1)
        new_active = (active_s | add) & ~drop
        active_s = torch.where(changed.unsqueeze(-1), new_active, active_s)
        if idx is None:
            active = active_s
            dirty = changed
        else:
            active.index_copy_(0, idx, active_s)
            dirty.zero_()
            dirty.index_copy_(0, idx, changed)
    ineq = (mv(g_mat, y) - h_b).clamp_min(0.0).amax(dim=-1)
    if neq > 0:
        eq = (mv(a_mat, y) - as_batch_vec(b, batch, neq, x)).abs().amax(dim=-1)
    else:
        eq = y.new_zeros(batch)
    dual_neg = (-lam).clamp_min(0.0).amax(dim=-1)
    failed = (
        dirty | (ineq > HYBRID_CV_TOL) | (eq > HYBRID_CV_TOL) | (dual_neg > HYBRID_CV_TOL)
    )
    return y, lam, nu, failed


def _pdipm_subset(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    idx: Tensor,
    eps: float,
    max_iter: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the PDIPM on a subset of the batch.

    Args:
        x: Full-batch points.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        idx: Rows to solve.
        eps: Interior-point residual tolerance.
        max_iter: Maximum Mehrotra iterations.

    Returns:
        Tuple ``(zhat, nu, lam, slack)`` for the selected rows.
    """
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return pdipm_forward(
        eye,
        -x.index_select(0, idx),
        _select_batch(g_mat, idx, 2),
        _select_batch(h, idx, 1),
        _select_batch(a_mat, idx, 2),
        _select_batch(b, idx, 1),
        eps=eps,
        max_iter=max_iter,
    )


def hybrid_forward(
    x: Tensor,
    a_mat: Tensor,
    b: Tensor,
    g_mat: Tensor,
    h: Tensor,
    *,
    n_admm: int = HYBRID_DEFAULT_N_ADMM,
    slack_tol: float = HYBRID_DEFAULT_SLACK_TOL,
    max_repair: int = HYBRID_DEFAULT_MAX_REPAIR,
    eps: float = 1e-12,
    max_iter: int = 20,
    sigma: float = Constants.PROJECTION_DEFAULT_SIGMA,
    omega: float = Constants.PROJECTION_DEFAULT_OMEGA,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project onto ``A y = b``, ``G y <= h`` at qpth accuracy.

    Args:
        x: Points to project, ``(B, n)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``.
        b: Equality right-hand side, ``(neq,)`` or ``(B, neq)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        h: Inequality right-hand side, ``(nineq,)`` or ``(B, nineq)``.
        n_admm: Douglas-Rachford iterations used as a warm start.
        slack_tol: Active-set slack threshold after ADMM.
        max_repair: Maximum active-set add/drop sweeps.
        eps: PDIPM residual tolerance used on rows that do not polish.
        max_iter: PDIPM iterations used on rows that do not polish.
        sigma: ADMM stepsize.
        omega: ADMM relaxation.

    Returns:
        Tuple ``(zhat, nu, lam, slack)``.
    """
    y0 = _affine_admm(x, a_mat, b, g_mat, h, n_admm, sigma, omega)
    y, lam, nu, failed = _active_set_polish(
        x, y0, a_mat, b, g_mat, h, slack_tol, max_repair
    )
    if bool(failed.any()):
        if bool(failed.all()):
            eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
            y, nu, lam, slack = pdipm_forward(
                eye, -x, g_mat, h, a_mat, b, eps=eps, max_iter=max_iter
            )
            return y, nu, lam, slack
        idx = failed.nonzero(as_tuple=True)[0]
        y_f, nu_f, lam_f, slack_f = _pdipm_subset(
            x, a_mat, b, g_mat, h, idx, eps, max_iter
        )
        y = y.clone()
        lam = lam.clone()
        nu = nu.clone()
        y.index_copy_(0, idx, y_f)
        lam.index_copy_(0, idx, lam_f)
        nu.index_copy_(0, idx, nu_f)
        slack = (
            as_batch_vec(h, x.shape[0], int(g_mat.shape[-2]), x) - mv(g_mat, y)
        ).clamp(min=1e-12)
        slack.index_copy_(0, idx, slack_f)
        return y, nu, lam, slack
    slack = (as_batch_vec(h, x.shape[0], int(g_mat.shape[-2]), x) - mv(g_mat, y)).clamp(
        min=1e-12
    )
    return y, nu, lam, slack
