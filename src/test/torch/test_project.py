"""Forward-pass parity between the Torch and JAX projection layers."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from pinet import (
    AffineInequalityConstraint as JaxIneq,
)
from pinet import (
    BoxConstraint as JaxBox,
)
from pinet import (
    BoxConstraintSpecification,
    NonLinearSpecification,
    ProjectionInstance,
    SOCType,
)
from pinet import (
    EqualityConstraint as JaxEq,
)
from pinet import (
    NonLinearConstraint as JaxNL,
)
from pinet import (
    Project as JaxProject,
)
from pinet.torch import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    NonLinearConstraint,
    Project,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [24, 42]


def _t(array: object) -> torch.Tensor:
    """Convert a JAX/numpy array to a float64 CPU tensor.

    Args:
        array: Array-like value.

    Returns:
        Float64 tensor.
    """
    return torch.tensor(np.asarray(array), dtype=torch.float64)


def _close(
    left: torch.Tensor, right: object, atol: float = 1e-4, rtol: float = 1e-4
) -> None:
    """Assert two arrays match.

    Args:
        left: Torch tensor.
        right: JAX or numpy array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    right_t = _t(right)
    assert torch.allclose(left, right_t, atol=atol, rtol=rtol), (
        f"Torch and JAX projections disagree. max abs "
        f"{(left - right_t).abs().max().item()}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_box_only_matches_jax(seed: int) -> None:
    """Box-only projection matches the JAX closed form."""
    rng = np.random.default_rng(seed)
    lb = np.array([[-np.inf], [0.0]], dtype=np.float64).reshape(1, 2, 1)
    ub = np.array([[0.0], [np.inf]], dtype=np.float64).reshape(1, 2, 1)
    x = rng.uniform(-2, 2, size=(5, 2, 1))

    jax_proj = JaxProject(
        box_constraint=JaxBox(
            BoxConstraintSpecification(lb=jnp.array(lb), ub=jnp.array(ub))
        )
    )
    torch_proj = Project(box_constraint=BoxConstraint(lb=lb, ub=ub), compile=False)
    y_jax = jax_proj.call(y_raw=ProjectionInstance(x=jnp.array(x)))[0].x
    y_torch = torch_proj(_t(x))
    _close(y_torch, y_jax)


@pytest.mark.parametrize("seed", SEEDS)
def test_eq_only_matches_jax(seed: int) -> None:
    """Equality-only projection matches the JAX closed form."""
    rng = np.random.default_rng(seed)
    dim = 8
    n_eq = 3
    a_mat = rng.normal(size=(1, n_eq, dim))
    b = a_mat @ rng.normal(size=(1, dim, 1))
    x = rng.normal(size=(4, dim, 1))

    jax_proj = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b), method="pinv")
    )
    torch_proj = Project(
        eq_constraint=EqualityConstraint(a_mat, b, method="pinv"), compile=False
    )
    y_jax = jax_proj.call(y_raw=ProjectionInstance(x=jnp.array(x)))[0].x
    y_torch = torch_proj(_t(x))
    _close(y_torch, y_jax)


@pytest.mark.parametrize("seed, batch_size", list(product(SEEDS, [1, 5])))
def test_eq_ineq_matches_jax(seed: int, batch_size: int) -> None:
    """ADMM polytope projection matches JAX for eq + ineq constraints."""
    rng = np.random.default_rng(seed)
    dim = 12
    n_eq = 4
    n_ineq = 5
    a_mat = rng.normal(size=(1, n_eq, dim))
    c_mat = rng.normal(size=(1, n_ineq, dim))
    x_feas = rng.uniform(-1, 1, size=(1, dim, 1))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.5
    ub = c_mat @ x_feas + 0.5
    x = rng.uniform(-2, 2, size=(batch_size, dim, 1))
    n_iter = 80

    jax_proj = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b), method="pinv"),
        ineq_constraint=JaxIneq(
            c_mat=jnp.array(c_mat), lb=jnp.array(lb), ub=jnp.array(ub)
        ),
    )
    torch_proj = Project(
        eq_constraint=EqualityConstraint(a_mat, b, method="pinv"),
        ineq_constraint=AffineInequalityConstraint(c_mat, lb, ub),
        n_iter=n_iter,
        compile=False,
    )
    y_jax = jax_proj.call(
        y_raw=ProjectionInstance(x=jnp.array(x)), n_iter=n_iter, sigma=1.0, omega=1.7
    )[0].x
    y_torch = torch_proj(_t(x), n_iter=n_iter, sigma=1.0, omega=1.7)
    _close(y_torch, y_jax)


@pytest.mark.parametrize("seed", SEEDS)
def test_eq_ineq_box_matches_jax(seed: int) -> None:
    """ADMM projection with a primal box matches JAX."""
    rng = np.random.default_rng(seed)
    dim = 10
    n_eq = 3
    n_ineq = 4
    a_mat = rng.normal(size=(1, n_eq, dim))
    c_mat = rng.normal(size=(1, n_ineq, dim))
    x_feas = rng.uniform(-0.5, 0.5, size=(1, dim, 1))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.25
    ub = c_mat @ x_feas + 0.25
    box_lb = np.full((1, dim, 1), -1.0)
    box_ub = np.full((1, dim, 1), 1.0)
    x = rng.uniform(-2, 2, size=(3, dim, 1))
    n_iter = 100

    jax_box = JaxBox(
        BoxConstraintSpecification(lb=jnp.array(box_lb), ub=jnp.array(box_ub))
    )
    jax_proj = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b), method="pinv"),
        ineq_constraint=JaxIneq(
            c_mat=jnp.array(c_mat), lb=jnp.array(lb), ub=jnp.array(ub)
        ),
        box_constraint=jax_box,
    )
    torch_proj = Project(
        EqualityConstraint(a_mat, b, method="pinv"),
        AffineInequalityConstraint(c_mat, lb, ub),
        BoxConstraint(lb=box_lb, ub=box_ub),
        n_iter=n_iter,
        compile=False,
    )
    y_jax = jax_proj.call(y_raw=ProjectionInstance(x=jnp.array(x)), n_iter=n_iter)[0].x
    y_torch = torch_proj(_t(x), n_iter=n_iter)
    _close(y_torch, y_jax)


@pytest.mark.parametrize("seed", SEEDS)
def test_soc_matches_jax(seed: int) -> None:
    """Lifted SOC projection matches JAX."""
    rng = np.random.default_rng(seed)
    dim = 6
    n_eq = 2
    n_soc = 3
    x_feas = rng.uniform(-1, 1, size=(1, dim, 1))
    a_mat = rng.uniform(-1, 1, size=(1, n_eq, dim))
    b = a_mat @ x_feas
    a_soc = rng.uniform(0.5, 1.5, size=(1, n_soc, dim))
    a_off = rng.uniform(0.1, 0.5, size=(1, n_soc, 1))
    f_soc = rng.uniform(0.0, 1.0, size=(1, 1, dim))
    b_soc = (
        0.1
        + np.linalg.norm(a_soc @ x_feas + a_off, axis=1, keepdims=True)
        - f_soc @ x_feas
    )
    x = rng.uniform(-3, 3, size=(2, dim, 1))
    n_iter = 200

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=jnp.array(a_soc),
        a=jnp.array(a_off),
        f=jnp.array(f_soc),
        b=jnp.array(b_soc),
    )
    jax_proj = JaxProject(
        eq_constraint=JaxEq(a_mat=jnp.array(a_mat), b=jnp.array(b)),
        nl_constraints=[JaxNL(spec=nl_spec)],
        unroll=True,
    )
    torch_proj = Project(
        eq_constraint=EqualityConstraint(a_mat, b),
        nl_constraints=[
            NonLinearConstraint(a_mat=a_soc, a=a_off, f=f_soc, b=b_soc, nl_type=SOCType)
        ],
        n_iter=n_iter,
        unroll=True,
        compile=False,
    )
    y_jax = jax_proj.call(
        y_raw=ProjectionInstance(x=jnp.array(x), nl=[nl_spec]),
        n_iter=n_iter,
        sigma=2.0,
        omega=1.8,
    )[0].x
    y_torch = torch_proj(_t(x), n_iter=n_iter, sigma=2.0, omega=1.8)
    _close(y_torch, y_jax, atol=1e-3, rtol=1e-3)


def test_row_layout_roundtrip() -> None:
    """``(B, n)`` inputs return ``(B, n)`` outputs."""
    lb = np.full((1, 3), -1.0)
    ub = np.full((1, 3), 1.0)
    x = torch.tensor([[2.0, -2.0, 0.5], [0.0, 3.0, -4.0]], dtype=torch.float64)
    proj = Project(box_constraint=BoxConstraint(lb=lb, ub=ub), compile=False)
    y = proj(x)
    assert y.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(y.shape)}"
    assert torch.allclose(y, y.clamp(-1, 1)), "Box projection should clip to [-1, 1]."
