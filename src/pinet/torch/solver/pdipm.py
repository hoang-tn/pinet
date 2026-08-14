"""Batched Mehrotra PDIPM for dense QPs, compatible with qpth.

qpth's CPU path spends most of its time completing a partial LU of the
Schur complement via ``torch.lu_unpack`` on every interior-point iteration.
This module uses the same predictor-corrector iteration, but refactors the
full Schur complement with ``torch.linalg.lu_factor`` and keeps shared
``Q``, ``G``, and ``A`` unbatched so those products are computed once.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SchurCache:
    """Cached products that do not depend on the interior-point slacks.

    Attributes:
        q_mat: Quadratic term, ``(n, n)`` or ``(B, n, n)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        q_lu: LU factors of ``Q``, or ``None`` when ``Q`` is the identity.
        a_qinv_at: ``A Q^{-1} A^T``.
        g_qinv_at: ``G Q^{-1} A^T``.
        g_qinv_gt: ``G Q^{-1} G^T``.
        identity_q: Whether ``Q`` is the identity.
        neq: Number of equalities.
        nineq: Number of inequalities.
        nz: Number of primal variables.
    """

    q_mat: Tensor
    a_mat: Tensor
    g_mat: Tensor
    q_lu: tuple[Tensor, Tensor] | None
    a_qinv_at: Tensor
    g_qinv_at: Tensor
    g_qinv_gt: Tensor
    identity_q: bool
    neq: int
    nineq: int
    nz: int


@dataclass
class SchurWorkspace:
    """Reusable Schur-complement storage for one forward or backward solve.

    Attributes:
        cache: Constant products.
        s_mat: Dense Schur matrix, shape ``(B, neq + nineq, neq + nineq)``.
        ggt_diag: Diagonal of ``G Q^{-1} G^T``.
        use_chol: Whether to factor with Cholesky (identity ``Q``).
        rhs: Scratch right-hand side, shape ``(B, neq + nineq)``.
    """

    cache: SchurCache
    s_mat: Tensor
    ggt_diag: Tensor
    use_chol: bool
    rhs: Tensor


@dataclass
class SchurFactor:
    """Factorization of the current Schur complement.

    Attributes:
        data: Cholesky factor, or packed LU.
        pivots: LU pivots, or ``None`` when ``data`` is a Cholesky factor.
    """

    data: Tensor
    pivots: Tensor | None


def mv(matrix: Tensor, vec: Tensor) -> Tensor:
    """Batched ``matrix @ vec`` with an unbatched or batched matrix.

    Args:
        matrix: ``(..., m, n)``.
        vec: ``(B, n)``.

    Returns:
        Product of shape ``(B, m)``.
    """
    return torch.matmul(matrix, vec.unsqueeze(-1)).squeeze(-1)


def mv_t(matrix: Tensor, vec: Tensor) -> Tensor:
    """Batched ``matrix.T @ vec``.

    Args:
        matrix: ``(..., m, n)``.
        vec: ``(B, m)``.

    Returns:
        Product of shape ``(B, n)``.
    """
    return torch.matmul(matrix.transpose(-1, -2), vec.unsqueeze(-1)).squeeze(-1)


def max_step(value: Tensor, delta: Tensor) -> Tensor:
    """Largest step in ``(0, 1]`` that keeps ``value + step * delta`` positive.

    Args:
        value: Positive primal or dual slack.
        delta: Search direction.

    Returns:
        Per-batch step lengths.
    """
    ratio = torch.where(delta < 0, -value / delta, 1.0)
    return ratio.amin(dim=-1).clamp(max=1.0)


def outer(left: Tensor, right: Tensor) -> Tensor:
    """Batched outer product.

    Args:
        left: ``(B, m)``.
        right: ``(B, n)``.

    Returns:
        ``(B, m, n)``.
    """
    return left.unsqueeze(-1) * right.unsqueeze(-2)


def batch_size(
    q_mat: Tensor,
    p: Tensor,
    g_mat: Tensor,
    h: Tensor,
    a_mat: Tensor,
    b: Tensor,
) -> int:
    """Infer the QP batch size from the first batched coefficient.

    Args:
        q_mat: Quadratic term.
        p: Linear term.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        a_mat: Equality matrix.
        b: Equality right-hand side.

    Returns:
        Batch size, or ``1`` when every coefficient is unbatched.
    """
    for tensor, ndim in (
        (p, 2),
        (h, 2),
        (b, 2),
        (q_mat, 3),
        (g_mat, 3),
        (a_mat, 3),
    ):
        if tensor.numel() > 0 and tensor.ndim == ndim:
            return int(tensor.shape[0])
    return 1


def as_batch_vec(tensor: Tensor, batch: int, length: int, ref: Tensor) -> Tensor:
    """Broadcast a vector coefficient to ``(batch, length)``.

    Args:
        tensor: ``(length,)`` or ``(B, length)``. Empty tensors become zeros.
        batch: Target batch size.
        length: Vector length.
        ref: Tensor used for dtype and device of an empty result.

    Returns:
        Broadcast vector.
    """
    if length == 0 or tensor.numel() == 0:
        return ref.new_zeros(batch, length)
    if tensor.ndim == 1:
        return tensor.reshape(1, length).expand(batch, length)
    return tensor.expand(batch, length)


def is_identity_q(q_mat: Tensor) -> bool:
    """Return whether ``q_mat`` is an unbatched identity.

    Args:
        q_mat: Quadratic term.

    Returns:
        ``True`` when a 2-D (or batch-1) ``Q`` equals the identity.
    """
    if q_mat.shape[-1] != q_mat.shape[-2]:
        return False
    q_work = q_mat[0] if q_mat.ndim == 3 and q_mat.shape[0] == 1 else q_mat
    if q_work.ndim != 2:
        return False
    eye = torch.eye(q_work.shape[0], dtype=q_work.dtype, device=q_work.device)
    return bool(torch.equal(q_work, eye))


def apply_qinv(cache: SchurCache, rhs: Tensor) -> Tensor:
    """Apply ``Q^{-1}`` to a batch of vectors.

    Args:
        cache: Precomputed factorization.
        rhs: Right-hand side ``(B, n)``.

    Returns:
        ``Q^{-1} rhs``.
    """
    if cache.q_lu is None:
        return rhs
    lu_data, pivots = cache.q_lu
    if lu_data.ndim == 2:
        return torch.linalg.lu_solve(lu_data, pivots, rhs.mT).mT
    return torch.linalg.lu_solve(lu_data, pivots, rhs.unsqueeze(-1)).squeeze(-1)


def apply_q(cache: SchurCache, rhs: Tensor) -> Tensor:
    """Apply ``Q`` to a batch of vectors.

    Args:
        cache: Precomputed factorization.
        rhs: Right-hand side ``(B, n)``.

    Returns:
        ``Q rhs``.
    """
    if cache.identity_q:
        return rhs
    return mv(cache.q_mat, rhs)


def precompute_schur(q_mat: Tensor, g_mat: Tensor, a_mat: Tensor) -> SchurCache:
    """Factor ``Q`` and cache the constant blocks of the Schur complement.

    Args:
        q_mat: Quadratic term.
        g_mat: Inequality matrix.
        a_mat: Equality matrix. May be empty.

    Returns:
        Cached products reused on every interior-point iteration.
    """
    nineq, nz = int(g_mat.shape[-2]), int(g_mat.shape[-1])
    if a_mat.numel() == 0:
        a_mat = g_mat.new_zeros(*g_mat.shape[:-2], 0, nz)
    neq = int(a_mat.shape[-2])
    identity_q = is_identity_q(q_mat)
    a_t = a_mat.transpose(-1, -2)
    g_t = g_mat.transpose(-1, -2)
    if identity_q:
        q_lu = None
        q_inv_at = a_t
        q_inv_gt = g_t
    else:
        q_lu = torch.linalg.lu_factor(q_mat)
        q_inv_at = torch.linalg.lu_solve(q_lu[0], q_lu[1], a_t)
        q_inv_gt = torch.linalg.lu_solve(q_lu[0], q_lu[1], g_t)
    return SchurCache(
        q_mat=q_mat,
        a_mat=a_mat,
        g_mat=g_mat,
        q_lu=q_lu,
        a_qinv_at=torch.matmul(a_mat, q_inv_at),
        g_qinv_at=torch.matmul(g_mat, q_inv_at),
        g_qinv_gt=torch.matmul(g_mat, q_inv_gt),
        identity_q=identity_q,
        neq=neq,
        nineq=nineq,
        nz=nz,
    )


def make_workspace(cache: SchurCache, batch: int) -> SchurWorkspace:
    """Allocate a Schur matrix and fill the blocks that do not depend on slacks.

    Args:
        cache: Constant Schur blocks.
        batch: Batch size.

    Returns:
        Workspace reused on every interior-point iteration.
    """
    kkt = cache.neq + cache.nineq
    s_mat = cache.g_qinv_gt.new_zeros(batch, kkt, kkt)
    if cache.neq > 0:
        s_mat[:, : cache.neq, : cache.neq] = cache.a_qinv_at
        a21 = cache.g_qinv_at.expand(batch, cache.nineq, cache.neq)
        s_mat[:, cache.neq :, : cache.neq] = a21
        s_mat[:, : cache.neq, cache.neq :] = a21.transpose(-1, -2)
    s_mat[:, cache.neq :, cache.neq :] = cache.g_qinv_gt
    return SchurWorkspace(
        cache=cache,
        s_mat=s_mat,
        ggt_diag=torch.diagonal(cache.g_qinv_gt, dim1=-2, dim2=-1),
        use_chol=cache.identity_q,
        rhs=cache.g_qinv_gt.new_empty(batch, kkt),
    )


def factor_schur(workspace: SchurWorkspace, dinv: Tensor) -> SchurFactor:
    """Factor the Schur complement with the current slack scaling.

    Args:
        workspace: Preallocated Schur matrix.
        dinv: ``s / z``, shape ``(B, nineq)``.

    Returns:
        Cholesky factor when ``Q`` is the identity, otherwise LU.
    """
    cache = workspace.cache
    workspace.s_mat[:, cache.neq :, cache.neq :].diagonal(dim1=-2, dim2=-1).copy_(
        workspace.ggt_diag + dinv
    )
    if workspace.use_chol:
        chol, info = torch.linalg.cholesky_ex(workspace.s_mat)
        if not bool(torch.any(info)):
            return SchurFactor(chol, None)
    lu_data, pivots = torch.linalg.lu_factor(workspace.s_mat)
    return SchurFactor(lu_data, pivots)


def solve_schur(factor: SchurFactor, rhs: Tensor) -> Tensor:
    """Solve ``S w = rhs`` with the current Schur factorization.

    Args:
        factor: Cholesky or LU factorization of ``S``.
        rhs: Right-hand side ``(B, neq + nineq)``.

    Returns:
        Solution ``w``.
    """
    rhs_col = rhs.unsqueeze(-1)
    if factor.pivots is None:
        return torch.cholesky_solve(rhs_col, factor.data).squeeze(-1)
    return torch.linalg.lu_solve(factor.data, factor.pivots, rhs_col).squeeze(-1)


def solve_kkt(
    workspace: SchurWorkspace,
    factor: SchurFactor,
    d_scale: Tensor,
    rx: Tensor,
    rs: Tensor,
    rz: Tensor,
    ry: Tensor,
    *,
    rx_zero: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve the reduced KKT system for one Newton step.

    Args:
        workspace: Preallocated Schur storage.
        factor: Factorization of the current Schur complement.
        d_scale: ``z / s``.
        rx: Stationarity residual.
        rs: Complementarity residual.
        rz: Inequality residual.
        ry: Equality residual.
        rx_zero: Skip ``Q^{-1} rx`` when the stationarity residual is zero.

    Returns:
        Tuple ``(dx, ds, dz, dy)``.
    """
    cache = workspace.cache
    rhs = workspace.rhs
    if rx_zero:
        rhs[:, : cache.neq] = 0
        rhs[:, cache.neq :] = rs / d_scale
        w = solve_schur(factor, -rhs)
        w_eq = w[:, : cache.neq]
        w_ineq = w[:, cache.neq :]
        g1 = -mv_t(cache.g_mat, w_ineq) - mv_t(cache.a_mat, w_eq)
        dx = apply_qinv(cache, g1)
        ds = (-rs - w_ineq) / d_scale
        return dx, ds, w_ineq, w_eq
    inv_q_rx = apply_qinv(cache, rx)
    rhs[:, : cache.neq] = mv(cache.a_mat, inv_q_rx) - ry
    rhs[:, cache.neq :] = mv(cache.g_mat, inv_q_rx) + rs / d_scale - rz
    w = solve_schur(factor, -rhs)
    w_eq = w[:, : cache.neq]
    w_ineq = w[:, cache.neq :]
    g1 = -rx - mv_t(cache.g_mat, w_ineq) - mv_t(cache.a_mat, w_eq)
    dx = apply_qinv(cache, g1)
    ds = (-rs - w_ineq) / d_scale
    return dx, ds, w_ineq, w_eq


def _shift_positive(values: Tensor) -> Tensor:
    """Shift a batch row so its minimum is at least 1 when it was negative.

    Args:
        values: Slack or dual vector ``(B, m)``.

    Returns:
        Shifted values.
    """
    min_val = values.amin(dim=-1, keepdim=True)
    return torch.where(min_val < 0, values - min_val + 1, values)


def _residuals(
    cache: SchurCache,
    x: Tensor,
    slack: Tensor,
    lam: Tensor,
    nu: Tensor,
    p: Tensor,
    h: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """KKT residuals and the Mehrotra duality measure.

    Args:
        cache: Constant Schur blocks.
        x: Primal iterate.
        slack: Inequality slack.
        lam: Inequality dual.
        nu: Equality dual.
        p: Linear term.
        h: Inequality right-hand side.
        b: Equality right-hand side.

    Returns:
        Tuple ``(rx, rz, ry, mu)``.
    """
    rx = apply_q(cache, x) + mv_t(cache.g_mat, lam) + mv_t(cache.a_mat, nu) + p
    rz = mv(cache.g_mat, x) + slack - h
    ry = mv(cache.a_mat, x) - b
    mu = (slack * lam).sum(dim=-1).abs() / cache.nineq
    return rx, rz, ry, mu


def _mehrotra_step(
    workspace: SchurWorkspace,
    x: Tensor,
    slack: Tensor,
    lam: Tensor,
    nu: Tensor,
    rx: Tensor,
    rz: Tensor,
    ry: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One predictor-corrector Newton step.

    Args:
        workspace: Preallocated Schur storage.
        x: Primal iterate.
        slack: Inequality slack.
        lam: Inequality dual.
        nu: Equality dual.
        rx: Stationarity residual.
        rz: Inequality residual.
        ry: Equality residual.
        mu: Duality measure.

    Returns:
        Updated ``(x, slack, lam, nu)``.
    """
    d_scale = lam / slack
    factor = factor_schur(workspace, slack / lam)
    dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
        workspace, factor, d_scale, rx, lam, rz, ry
    )
    alpha = torch.minimum(max_step(lam, dz_aff), max_step(slack, ds_aff))
    alpha_ineq = alpha.unsqueeze(-1)
    sig = (
        ((slack + alpha_ineq * ds_aff) * (lam + alpha_ineq * dz_aff)).sum(dim=-1)
        / (slack * lam).sum(dim=-1)
    ).pow(3)
    rs_cor = ((-mu * sig).unsqueeze(-1) + ds_aff * dz_aff) / slack
    dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
        workspace,
        factor,
        d_scale,
        rx,
        rs_cor,
        rz,
        ry,
        rx_zero=True,
    )
    dx = dx_aff + dx_cor
    ds = ds_aff + ds_cor
    dz = dz_aff + dz_cor
    dy = dy_aff + dy_cor
    step = (0.999 * torch.minimum(max_step(lam, dz), max_step(slack, ds))).unsqueeze(-1)
    return x + step * dx, slack + step * ds, lam + step * dz, nu + step * dy


def pdipm_forward(
    q_mat: Tensor,
    p: Tensor,
    g_mat: Tensor,
    h: Tensor,
    a_mat: Tensor,
    b: Tensor,
    *,
    eps: float = 1e-12,
    max_iter: int = 20,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve a batch of dense QPs with a Mehrotra primal-dual IPM.

    The problem for each batch item is

    .. math::

        \\min_z \\tfrac{1}{2} z^T Q z + p^T z
        \\quad \\text{s.t.} \\quad G z \\le h,\\; A z = b.

    Args:
        q_mat: Quadratic term, ``(n, n)`` or ``(B, n, n)``.
        p: Linear term, ``(n,)`` or ``(B, n)``.
        g_mat: Inequality matrix, ``(nineq, n)`` or ``(B, nineq, n)``.
        h: Inequality right-hand side, ``(nineq,)`` or ``(B, nineq)``.
        a_mat: Equality matrix, ``(neq, n)`` or ``(B, neq, n)``. Empty if
            there are no equalities.
        b: Equality right-hand side, ``(neq,)`` or ``(B, neq)``.
        eps: Stop when every batch residual is below this value.
        max_iter: Maximum predictor-corrector iterations.

    Returns:
        Tuple ``(zhat, nu, lam, slack)``.

    Raises:
        ValueError: If there are no inequality constraints.
    """
    cache = precompute_schur(q_mat, g_mat, a_mat)
    if cache.nineq <= 0:
        raise ValueError("pdipm_forward requires at least one inequality constraint.")
    n_batch = batch_size(q_mat, p, g_mat, h, a_mat, b)
    p = as_batch_vec(p, n_batch, cache.nz, g_mat)
    h = as_batch_vec(h, n_batch, cache.nineq, g_mat)
    b = as_batch_vec(b, n_batch, cache.neq, g_mat)
    workspace = make_workspace(cache, n_batch)

    d_scale = p.new_ones(n_batch, cache.nineq)
    factor = factor_schur(workspace, d_scale)
    x, slack, lam, nu = solve_kkt(
        workspace,
        factor,
        d_scale,
        p,
        p.new_zeros(n_batch, cache.nineq),
        -h,
        -b,
    )
    slack = _shift_positive(slack)
    lam = _shift_positive(lam)

    best_x, best_slack, best_lam, best_nu = x, slack, lam, nu
    best_resids = x.new_full((n_batch,), torch.inf)
    for _ in range(max_iter):
        rx, rz, ry, mu = _residuals(cache, x, slack, lam, nu, p, h, b)
        resids = (
            torch.linalg.vector_norm(rz, dim=-1)
            + torch.linalg.vector_norm(ry, dim=-1)
            + torch.linalg.vector_norm(rx, dim=-1)
            + cache.nineq * mu
        )
        improved = resids < best_resids
        best_resids = torch.where(improved, resids, best_resids)
        mask = improved.unsqueeze(-1)
        best_x = torch.where(mask, x, best_x)
        best_slack = torch.where(mask, slack, best_slack)
        best_lam = torch.where(mask, lam, best_lam)
        best_nu = torch.where(mask, nu, best_nu)
        if bool(best_resids.max() < eps):
            break
        x, slack, lam, nu = _mehrotra_step(workspace, x, slack, lam, nu, rx, rz, ry, mu)
    return best_x, best_nu, best_lam, best_slack
