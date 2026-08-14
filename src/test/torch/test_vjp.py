"""Gradient checks for the Torch projection layer vs finite differences and JAX."""

from collections.abc import Callable
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import Tensor

from pinet import AffineInequalityConstraint as JaxIneq
from pinet import EqualityConstraint as JaxEq
from pinet import Project as JaxProject
from pinet import ProjectionInstance
from pinet.torch import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    Project,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [8, 24]


def _t(array: object) -> Tensor:
    """Convert an array to a float64 tensor.

    Args:
        array: Array-like value.

    Returns:
        Float64 tensor.
    """
    return torch.tensor(np.asarray(array), dtype=torch.float64)


def _torch_loss(proj: Project, x: Tensor, vec: Tensor, **kwargs: Any) -> Tensor:
    """Inner-product loss used by the gradient checks.

    Args:
        proj: Torch projector.
        x: Point of shape ``(B, n)``.
        vec: Direction of shape ``(n, B)``.
        **kwargs: Forward overrides.

    Returns:
        Scalar loss.
    """
    y = proj(x, **kwargs)
    assert isinstance(y, Tensor)
    if y.ndim == 3:
        y = y.squeeze(-1)
    if vec.ndim == 1:
        return (y * vec).sum(dim=-1).mean()
    return (y * vec.T).sum(dim=-1).mean()


def _fd_directional(
    loss_at: Callable[[Tensor], Tensor],
    x: Tensor,
    direction: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Central finite-difference directional derivative.

    Args:
        loss_at: Callable mapping a point to a scalar tensor.
        x: Point.
        direction: Perturbation direction.
        eps: Step size.

    Returns:
        Directional derivative.
    """
    plus = loss_at(x + eps * direction)
    minus = loss_at(x - eps * direction)
    return (plus - minus) / (2 * eps)


@pytest.mark.parametrize("fpi", [True, False])
def test_triangle_jacobian(fpi: bool) -> None:
    """Analytic Jacobian of projection onto a triangle matches Torch IFT."""
    xs_bl = [
        np.array([[-0.5, -0.5]]),
        np.array([[-0.5, -0.25]]),
        np.array([[-0.25, -0.5]]),
    ]
    j_bl = np.zeros((1, 2, 2))
    xs_in = [np.array([[0.5, 0.25]]), np.array([[0.75, 0.5]])]
    j_in = np.eye(2).reshape(1, 2, 2)
    xs_b = [np.array([[0.5, -0.5]]), np.array([[0.25, -0.5]])]
    j_b = np.array([[[1.0, 0.0], [0.0, 0.0]]])

    box = BoxConstraint(
        lb=np.array([[[-np.inf], [0.0]]]),
        ub=np.array([[[1.0], [np.inf]]]),
    )
    ineq = AffineInequalityConstraint(
        c_mat=np.array([[[-1.0, 1.0]]]),
        lb=np.array([[[-np.inf]]]),
        ub=np.zeros((1, 1, 1)),
    )
    proj = Project(
        ineq_constraint=ineq,
        box_constraint=box,
        unroll=False,
        n_iter=100,
        n_iter_bwd=100,
        compile=False,
        fpi=fpi,
    )

    def jacobian_at(x_np: np.ndarray) -> Tensor:
        x = _t(x_np)
        rows: list[Tensor] = []
        for e in (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ):
            x_var = x.detach().clone().requires_grad_(True)
            loss = _torch_loss(
                proj, x_var, e.reshape(2, 1), sigma=1.0, omega=1.0, n_iter=100, fpi=fpi
            )
            (grad,) = torch.autograd.grad(loss, x_var)
            rows.append(grad.reshape(1, 1, 2))
        return torch.cat(rows, dim=1)

    for xs, j_true in ((xs_bl, j_bl), (xs_in, j_in), (xs_b, j_b)):
        for x in xs:
            jacobian = jacobian_at(x)
            assert torch.allclose(jacobian, _t(j_true), atol=1e-4, rtol=1e-4), (
                f"J={jacobian}, J_true={j_true} for x={x}"
            )


def test_box_jacobian() -> None:
    """Analytic Jacobian of a 2D box projection matches Torch autograd."""
    xs_tl = [np.array([[-0.5, 0.5]]), np.array([[-0.25, 0.5]])]
    j_tl = np.eye(2).reshape(1, 2, 2)
    xs_br = [np.array([[0.5, -0.5]]), np.array([[0.25, -0.5]])]
    j_br = np.zeros((1, 2, 2))
    proj = Project(
        box_constraint=BoxConstraint(
            lb=np.array([[[-np.inf], [0.0]]]),
            ub=np.array([[[0.0], [np.inf]]]),
        ),
        compile=False,
    )

    def jacobian_at(x_np: np.ndarray) -> Tensor:
        x = _t(x_np)
        rows: list[Tensor] = []
        for e in (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ):
            x_var = x.detach().clone().requires_grad_(True)
            loss = _torch_loss(proj, x_var, e)
            (grad,) = torch.autograd.grad(loss, x_var)
            rows.append(grad.reshape(1, 1, 2))
        return torch.cat(rows, dim=1)

    for xs, j_true in ((xs_tl, j_tl), (xs_br, j_br)):
        for x in xs:
            jacobian = jacobian_at(x)
            assert torch.allclose(jacobian, _t(j_true), atol=1e-4, rtol=1e-4), (
                f"J={jacobian}, J_true={j_true} for x={x}"
            )


@pytest.mark.parametrize("seed, batch_size", list(product(SEEDS, [1, 4])))
def test_eq_ineq_grad_matches_fd_and_jax(seed: int, batch_size: int) -> None:
    """Unroll, FPI, and BiCGSTAB grads match finite differences and JAX."""
    rng = np.random.default_rng(seed)
    dim = 16
    n_eq = 6
    n_ineq = 5
    a_mat = rng.normal(size=(1, n_eq, dim))
    c_mat = rng.normal(size=(1, n_ineq, dim))
    x_feas = rng.uniform(-1, 1, size=(1, dim, 1))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.4
    ub = c_mat @ x_feas + 0.4
    x_np = rng.uniform(-2, 2, size=(batch_size, dim))
    vec_np = rng.normal(size=(dim, batch_size))
    direction_np = rng.uniform(-1, 1, size=(batch_size, dim))
    n_iter = 250
    n_iter_bwd = 150
    sigma = 1.0
    omega = 1.7

    jax_unroll = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b), method="pinv"),
        ineq_constraint=JaxIneq(
            c_mat=jnp.array(c_mat), lb=jnp.array(lb), ub=jnp.array(ub)
        ),
        unroll=True,
    )
    jax_impl = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b), method="pinv"),
        ineq_constraint=JaxIneq(
            c_mat=jnp.array(c_mat), lb=jnp.array(lb), ub=jnp.array(ub)
        ),
        unroll=False,
    )

    def jax_loss(x_arr: jnp.ndarray, unroll: bool, fpi: bool) -> jnp.ndarray:
        layer = jax_unroll if unroll else jax_impl
        if unroll:
            y = layer.call(
                y_raw=ProjectionInstance(x=x_arr[..., None]),
                n_iter=n_iter,
                sigma=sigma,
                omega=omega,
            )[0].x[..., 0]
        else:
            y = layer.call(
                y_raw=ProjectionInstance(x=x_arr[..., None]),
                n_iter=n_iter,
                n_iter_bwd=n_iter_bwd,
                fpi=fpi,
                sigma=sigma,
                omega=omega,
            )[0].x[..., 0]
        return (y * jnp.array(vec_np).T).sum(axis=-1).mean()

    torch_unroll = Project(
        EqualityConstraint(a_mat, b, method="pinv"),
        AffineInequalityConstraint(c_mat, lb, ub),
        unroll=True,
        n_iter=n_iter,
        compile=False,
    )
    torch_impl = Project(
        EqualityConstraint(a_mat, b, method="pinv"),
        AffineInequalityConstraint(c_mat, lb, ub),
        unroll=False,
        n_iter=n_iter,
        n_iter_bwd=n_iter_bwd,
        compile=False,
    )

    x = _t(x_np)
    vec = _t(vec_np)
    direction = _t(direction_np)

    def grad_torch(proj: Project, fpi: bool | None = None) -> Tensor:
        x_var = x.detach().clone().requires_grad_(True)
        kwargs: dict[str, object] = {"sigma": sigma, "omega": omega, "n_iter": n_iter}
        if fpi is not None:
            kwargs["fpi"] = fpi
            kwargs["n_iter_bwd"] = n_iter_bwd
        loss = _torch_loss(proj, x_var, vec, **kwargs)
        (grad,) = torch.autograd.grad(loss, x_var)
        return grad

    g_unroll = grad_torch(torch_unroll)
    g_fpi = grad_torch(torch_impl, fpi=True)
    g_bicg = grad_torch(torch_impl, fpi=False)
    assert torch.allclose(g_unroll, g_fpi, atol=1e-4, rtol=1e-4), (
        "Unroll and FPI gradients disagree."
    )
    assert torch.allclose(g_unroll, g_bicg, atol=1e-4, rtol=1e-4), (
        "Unroll and BiCGSTAB gradients disagree."
    )

    def loss_unroll(point: Tensor) -> Tensor:
        return _torch_loss(
            torch_unroll, point, vec, sigma=sigma, omega=omega, n_iter=n_iter
        )

    fd = _fd_directional(loss_unroll, x, direction)
    for name, grad in (
        ("unroll", g_unroll),
        ("fpi", g_fpi),
        ("bicgstab", g_bicg),
    ):
        dirgrad = torch.sum(grad * direction)
        assert torch.allclose(dirgrad, fd, atol=1e-3, rtol=1e-3), (
            f"Finite-difference mismatch for {name}: "
            f"dir={float(dirgrad.detach())}, fd={float(fd.detach())}"
        )

    g_jax = jax.grad(lambda xx: jax_loss(xx, True, True))(jnp.array(x_np))
    assert torch.allclose(g_unroll, _t(g_jax), atol=1e-4, rtol=1e-4), (
        "Torch unroll gradient disagrees with JAX."
    )
    g_jax_impl = jax.grad(lambda xx: jax_loss(xx, False, False))(jnp.array(x_np))
    assert torch.allclose(g_bicg, _t(g_jax_impl), atol=1e-3, rtol=1e-3), (
        "Torch BiCGSTAB gradient disagrees with JAX."
    )
