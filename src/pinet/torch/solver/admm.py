"""ADMM / Douglas-Rachford iteration kernels (Torch)."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from pinet.torch._ops import box_project, equality_project, soc_project

AdmmLoop = Callable[..., tuple[Tensor, Tensor]]
SocPack = Sequence[Tensor]


def iteration_step(
    sk: Tensor,
    y: Tensor,
    a_mat: Tensor,
    a_pinv: Tensor,
    b: Tensor,
    lb: Tensor,
    ub: Tensor,
    scale: Tensor,
    sigma: Tensor,
    omega: Tensor,
    dim: int,
    n_soc: int,
    eps: Tensor,
    soc_pack: SocPack,
) -> Tensor:
    """One Douglas-Rachford / ADMM iteration.

    Args:
        sk: Governing sequence, shape ``(B, dim_lifted, 1)``.
        y: Point to project, shape ``(B, dim, 1)``.
        a_mat: Lifted equality matrix.
        a_pinv: Pseudoinverse of ``a_mat``.
        b: Lifted equality right-hand side.
        lb: Dense box lower bound.
        ub: Dense box upper bound.
        scale: Column scaling on the original coordinates.
        sigma: ADMM stepsize.
        omega: Relaxation parameter.
        dim: Original primal dimension.
        n_soc: Number of SOC cones.
        eps: SOC stabilizer.
        soc_pack: Flattened SOC tensors ``(mask_u, mask_t, a_full, b_full)*n_soc``.

    Returns:
        Next governing-sequence value.
    """
    zk = equality_project(sk, a_mat, a_pinv, b)
    reflect = 2 * zk - sk
    tobox_primal = (2 * sigma * scale * y + reflect[:, :dim, :]) / (
        1 + 2 * sigma * scale * scale
    )
    tobox = torch.cat([tobox_primal, reflect[:, dim:, :]], dim=1)
    tk = box_project(tobox, lb, ub)
    for i in range(n_soc):
        base = 4 * i
        tk = soc_project(
            tk,
            soc_pack[base],
            soc_pack[base + 1],
            soc_pack[base + 2],
            soc_pack[base + 3],
            eps,
        )
    return sk + omega * (tk - zk)


def admm_loop(
    n_iter: int,
    dim: int,
    n_soc: int,
    s0: Tensor,
    y: Tensor,
    a_mat: Tensor,
    a_pinv: Tensor,
    b: Tensor,
    lb: Tensor,
    ub: Tensor,
    scale: Tensor,
    d_c: Tensor,
    sigma: Tensor,
    omega: Tensor,
    eps: Tensor,
    soc_pack: SocPack,
) -> tuple[Tensor, Tensor]:
    """Run ``n_iter`` ADMM steps and return the unscaled primal slice.

    Args:
        n_iter: Number of Douglas-Rachford iterations.
        dim: Original primal dimension.
        n_soc: Number of SOC cones.
        s0: Initial governing sequence.
        y: Point to project.
        a_mat: Lifted equality matrix.
        a_pinv: Pseudoinverse of ``a_mat``.
        b: Lifted equality right-hand side.
        lb: Dense box lower bound.
        ub: Dense box upper bound.
        scale: Column scaling on the original coordinates.
        d_c: Column scaling used to unscale the output.
        sigma: ADMM stepsize.
        omega: Relaxation parameter.
        eps: SOC stabilizer.
        soc_pack: Flattened SOC tensors.

    Returns:
        Pair ``(projected_point, governing_sequence)``.
    """
    sk = s0
    for _ in range(n_iter):
        sk = iteration_step(
            sk,
            y,
            a_mat,
            a_pinv,
            b,
            lb,
            ub,
            scale,
            sigma,
            omega,
            dim,
            n_soc,
            eps,
            soc_pack,
        )
    zk = equality_project(sk, a_mat, a_pinv, b)
    y_out = zk[:, :dim, :] * d_c
    return y_out, sk


def make_admm_loop(n_iter: int, dim: int, n_soc: int) -> AdmmLoop:
    """Build an ADMM loop with static ``n_iter``, ``dim``, and ``n_soc``.

    Args:
        n_iter: Number of iterations, specialized into the loop.
        dim: Original primal dimension.
        n_soc: Number of SOC cones.

    Returns:
        Callable ``(s0, y, a_mat, a_pinv, b, lb, ub, scale, d_c, sigma, omega,
        eps, *soc_pack) -> (y_out, sK)``.
    """

    def _loop(
        s0: Tensor,
        y: Tensor,
        a_mat: Tensor,
        a_pinv: Tensor,
        b: Tensor,
        lb: Tensor,
        ub: Tensor,
        scale: Tensor,
        d_c: Tensor,
        sigma: Tensor,
        omega: Tensor,
        eps: Tensor,
        *soc_pack: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return admm_loop(
            n_iter,
            dim,
            n_soc,
            s0,
            y,
            a_mat,
            a_pinv,
            b,
            lb,
            ub,
            scale,
            d_c,
            sigma,
            omega,
            eps,
            soc_pack,
        )

    return _loop
