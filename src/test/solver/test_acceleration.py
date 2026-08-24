"""Tests for Anderson acceleration and residual-balancing of the ADMM solver."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)
from pinet.solver.acceleration import adapt_sigma, anderson_mix, init_accel_carry

jax.config.update("jax_enable_x64", True)


def _eq_ineq_layer(
    seed: int = 0,
    batch_size: int = 4,
    dim: int = 16,
    n_eq: int = 6,
    n_ineq: int = 8,
    **project_kwargs: Any,
) -> tuple[Project, ProjectionInstance]:
    """Build a random feasible equality-plus-inequality projection problem."""
    key = jax.random.PRNGKey(seed)
    ka_mat, kc_mat, kx, kfeas = jax.random.split(key, 4)
    a_mat = jax.random.normal(ka_mat, (1, n_eq, dim))
    x_feas = jax.random.normal(kfeas, (1, dim, 1))
    b = a_mat @ x_feas
    c_mat = jax.random.normal(kc_mat, (1, n_ineq, dim))
    slack = 0.5
    lb = c_mat @ x_feas - slack
    ub = c_mat @ x_feas + slack
    xinfeas = jax.random.normal(kx, (batch_size, dim, 1))
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)
    ineq = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)
    layer = Project(eq_constraint=eq, ineq_constraint=ineq, **project_kwargs)
    y_raw = ProjectionInstance(x=xinfeas)
    return layer, y_raw


def test_adapt_sigma_increases_when_primal_dominates() -> None:
    """Penalty grows when the consensus residual dominates the dual residual."""
    sigma = jnp.asarray(1.0)
    updated = adapt_sigma(
        sigma, primal_residual=jnp.asarray(20.0), dual_residual=jnp.asarray(1.0)
    )
    assert updated > sigma, (
        f"Expected sigma to increase when the primal residual dominates, got {updated}."
    )


def test_adapt_sigma_decreases_when_dual_dominates() -> None:
    """Penalty shrinks when the dual residual dominates the consensus residual."""
    sigma = jnp.asarray(1.0)
    updated = adapt_sigma(
        sigma, primal_residual=jnp.asarray(1.0), dual_residual=jnp.asarray(20.0)
    )
    assert updated < sigma, (
        f"Expected sigma to decrease when the dual residual dominates, got {updated}."
    )


def test_adapt_sigma_holds_when_residuals_are_balanced() -> None:
    """Penalty is unchanged when residuals are within the threshold ratio."""
    sigma = jnp.asarray(1.0)
    updated = adapt_sigma(
        sigma, primal_residual=jnp.asarray(2.0), dual_residual=jnp.asarray(1.0)
    )
    assert updated == sigma, (
        f"Expected sigma to stay put for balanced residuals, got {updated}."
    )


def test_anderson_mix_is_identity_at_a_fixed_point() -> None:
    """A zero residual history must return the unaccelerated operator output."""
    memory, batch, dim = 5, 3, 4
    x_hist = jnp.zeros((memory, batch, dim, 1))
    f_hist = jnp.zeros((memory, batch, dim, 1))
    g_last = jax.random.normal(jax.random.PRNGKey(1), (batch, dim, 1))
    mixed = anderson_mix(x_hist, f_hist, jnp.int32(memory), g_last)
    assert jnp.allclose(mixed, g_last), (
        "Anderson mixing of a zero residual must return g_last. "
        f"Expected {g_last}, got {mixed}."
    )


def test_anderson_mix_skips_short_history() -> None:
    """Fewer than two filled slots must return the unaccelerated output."""
    memory, batch, dim = 5, 2, 3
    x_hist = jax.random.normal(jax.random.PRNGKey(2), (memory, batch, dim, 1))
    f_hist = jax.random.normal(jax.random.PRNGKey(3), (memory, batch, dim, 1))
    g_last = jax.random.normal(jax.random.PRNGKey(4), (batch, dim, 1))
    mixed = anderson_mix(x_hist, f_hist, jnp.int32(1), g_last)
    assert jnp.allclose(mixed, g_last), (
        "Anderson mixing with a short history must return g_last. "
        f"Expected {g_last}, got {mixed}."
    )


def test_anderson_memory_must_be_at_least_two() -> None:
    """Constructing Project with Anderson memory 1 must raise."""
    a_mat = jnp.ones((1, 1, 2))
    b = jnp.ones((1, 1, 1))
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv")
    with pytest.raises(ValueError, match="anderson_memory"):
        Project(eq_constraint=eq, use_anderson=True, anderson_memory=1)


def test_accelerated_call_matches_original_at_convergence() -> None:
    """Anderson plus residual balancing must reach the same projection."""
    original, y_raw = _eq_ineq_layer(
        seed=11, use_anderson=False, use_adaptive_penalty=False
    )
    accelerated, _ = _eq_ineq_layer(
        seed=11, use_anderson=True, use_adaptive_penalty=True
    )
    y_orig = original.call(y_raw=y_raw, n_iter=400, sigma=1.0, omega=1.7)[0].x
    y_acc = accelerated.call(y_raw=y_raw, n_iter=400, sigma=1.0, omega=1.7)[0].x
    assert jnp.allclose(y_orig, y_acc, atol=1e-4, rtol=1e-4), (
        "Accelerated and original solvers must agree at convergence. "
        f"Max abs diff {jnp.max(jnp.abs(y_orig - y_acc))}."
    )


def test_call_and_check_accelerated_uses_fewer_or_equal_iterations() -> None:
    """Solve-to-tolerance with acceleration must not take more iterations."""
    layer, y_raw = _eq_ineq_layer(seed=21, batch_size=8, dim=24, n_eq=8, n_ineq=12)
    original = layer.call_and_check(
        sigma=1.0,
        omega=1.7,
        check_every=10,
        tol=1e-4,
        max_iter=400,
        reduction="max",
        use_anderson=False,
        use_adaptive_penalty=False,
    )
    accelerated = layer.call_and_check(
        sigma=1.0,
        omega=1.7,
        check_every=10,
        tol=1e-4,
        max_iter=400,
        reduction="max",
        use_anderson=True,
        use_adaptive_penalty=True,
    )
    y_orig, flag_orig, iters_orig = original(y_raw)
    y_acc, flag_acc, iters_acc = accelerated(y_raw)
    assert flag_orig, "Original solver should reach the requested tolerance."
    assert flag_acc, "Accelerated solver should reach the requested tolerance."
    assert iters_acc <= iters_orig, (
        "Accelerated call_and_check should use no more iterations than the "
        f"original loop. Original {iters_orig}, accelerated {iters_acc}."
    )
    assert jnp.allclose(y_orig.x, y_acc.x, atol=5e-4, rtol=5e-4), (
        "Both solvers must return equivalent projections. "
        f"Max abs diff {jnp.max(jnp.abs(y_orig.x - y_acc.x))}."
    )


def test_init_accel_carry_shapes() -> None:
    """Acceleration carry history must match the governing-sequence shape."""
    x = jnp.zeros((2, 5, 1))
    carry = init_accel_carry(ProjectionInstance(x=x), sigma=0.5, memory=4)
    assert carry.x_hist.shape == (4, 2, 5, 1), (
        f"Expected history shape (4, 2, 5, 1), got {carry.x_hist.shape}."
    )
    assert carry.n_filled == 0, (
        f"A fresh carry must start with empty history, got {carry.n_filled}."
    )


def test_accelerated_loop_is_jittable() -> None:
    """The accelerated Project.call path must compile under jax.jit."""
    layer, y_raw = _eq_ineq_layer(
        seed=3, use_anderson=True, use_adaptive_penalty=True
    )
    y = layer.call(y_raw=y_raw, n_iter=20, sigma=1.0, omega=1.7)[0].x
    assert y.shape == y_raw.x.shape, (
        f"Projected shape {y.shape} must match the input shape {y_raw.x.shape}."
    )
