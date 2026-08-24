"""Anderson acceleration and residual-balancing helpers for the ADMM solver."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from pinet._typing import BatchedLifted, ScalarLike
from pinet.constants import Constants
from pinet.dataclasses import ProjectionInstance

AdmmRawStep = Callable[
    [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
    tuple[ProjectionInstance, BatchedLifted, BatchedLifted],
]

ANDERSON_MEMORY = Constants.PROJECTION_ANDERSON_MEMORY
ANDERSON_MIN_HISTORY = Constants.PROJECTION_ANDERSON_MIN_HISTORY
ANDERSON_REGULARIZATION = Constants.PROJECTION_ANDERSON_REGULARIZATION
ANDERSON_SAFEGUARD = Constants.PROJECTION_ANDERSON_SAFEGUARD
ANDERSON_EVERY = Constants.PROJECTION_ANDERSON_EVERY
ANDERSON_STOP_RESIDUAL = Constants.PROJECTION_ANDERSON_STOP_RESIDUAL
ADAPTIVE_MU = Constants.PROJECTION_ADAPTIVE_PENALTY_MU
ADAPTIVE_TAU = Constants.PROJECTION_ADAPTIVE_PENALTY_TAU
ADAPTIVE_EVERY = Constants.PROJECTION_ADAPTIVE_PENALTY_EVERY
SIGMA_MIN = Constants.PROJECTION_ADAPTIVE_PENALTY_SIGMA_MIN
SIGMA_MAX = Constants.PROJECTION_ADAPTIVE_PENALTY_SIGMA_MAX


class AccelCarry(NamedTuple):
    """State carried by the accelerated Douglas-Rachford loop.

    Attributes:
        s: Governing sequence iterate.
        sigma: Current ADMM penalty.
        z_prev: Equality projection from the previous iteration.
        x_hist: Window of past governing-sequence iterates.
        f_hist: Window of past fixed-point residuals ``T(s) - s``.
        n_filled: Number of valid history slots (capped at memory).
        iter_idx: Number of iterations already performed.
    """

    s: ProjectionInstance
    sigma: Float[Array, ""]
    z_prev: BatchedLifted
    x_hist: Float[Array, "m B d_lifted 1"]
    f_hist: Float[Array, "m B d_lifted 1"]
    n_filled: Integer[Array, ""]
    iter_idx: Integer[Array, ""]


class AdaptiveCarry(NamedTuple):
    """State carried by residual-balancing without Anderson history.

    Attributes:
        s: Governing sequence iterate.
        sigma: Current ADMM penalty.
        z_prev: Equality projection from the previous iteration.
        iter_idx: Number of iterations already performed.
    """

    s: ProjectionInstance
    sigma: Float[Array, ""]
    z_prev: BatchedLifted
    iter_idx: Integer[Array, ""]


def init_adaptive_carry(
    s0: ProjectionInstance,
    sigma: float | Float[Array, ""],
) -> AdaptiveCarry:
    """Initialize a residual-balancing carry with no Anderson history.

    Args:
        s0: Initial governing sequence.
        sigma: Initial ADMM penalty.

    Returns:
        Carry ready for ``adaptive_loop``.
    """
    return AdaptiveCarry(
        s=s0,
        sigma=jnp.asarray(sigma, dtype=s0.x.dtype),
        z_prev=jnp.zeros_like(s0.x),
        iter_idx=jnp.array(0, dtype=jnp.int32),
    )


def init_accel_carry(
    s0: ProjectionInstance,
    sigma: float | Float[Array, ""],
    memory: int,
) -> AccelCarry:
    """Initialize the accelerated-loop carry from a governing sequence.

    Args:
        s0: Initial governing sequence.
        sigma: Initial ADMM penalty.
        memory: Anderson history length.

    Returns:
        Carry with empty Anderson history.
    """
    zeros = jnp.zeros((memory, *s0.x.shape), dtype=s0.x.dtype)
    return AccelCarry(
        s=s0,
        sigma=jnp.asarray(sigma, dtype=s0.x.dtype),
        z_prev=jnp.zeros_like(s0.x),
        x_hist=zeros,
        f_hist=zeros,
        n_filled=jnp.array(0, dtype=jnp.int32),
        iter_idx=jnp.array(0, dtype=jnp.int32),
    )


def _mean_norm(residual: BatchedLifted) -> Float[Array, ""]:
    """Mean Euclidean residual over the batch.

    Args:
        residual: Per-sample residual tensors.

    Returns:
        Mean of the Euclidean norms.
    """
    flat = jnp.asarray(residual).reshape(residual.shape[0], -1)
    return jnp.mean(jnp.linalg.norm(flat, axis=-1))


def adapt_sigma(
    sigma: Float[Array, ""],
    primal_residual: Float[Array, ""],
    dual_residual: Float[Array, ""],
    mu: float = ADAPTIVE_MU,
    tau: float = ADAPTIVE_TAU,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
) -> Float[Array, ""]:
    """Residual-balance the ADMM penalty.

    Increases ``sigma`` when the primal (consensus) residual dominates and
    decreases it when the dual residual dominates, following Boyd et al.
    In this Douglas-Rachford splitting ``sigma`` plays the role of
    :math:`\\rho / 2`, so a larger penalty pulls the two block-projections
    together.

    Args:
        sigma: Current penalty.
        primal_residual: Consensus residual ``||t - z||``.
        dual_residual: Equality-block movement ``||z - z_prev||``.
        mu: Residual-ratio threshold that triggers an update.
        tau: Multiplicative penalty step.
        sigma_min: Lower bound on the adapted penalty.
        sigma_max: Upper bound on the adapted penalty.

    Returns:
        Updated penalty, clipped to ``[sigma_min, sigma_max]``.
    """
    increased = sigma * tau
    decreased = sigma / tau
    too_primal = primal_residual > mu * dual_residual
    too_dual = dual_residual > mu * primal_residual
    sigma_new = jnp.where(too_primal, increased, jnp.where(too_dual, decreased, sigma))
    return jnp.clip(sigma_new, sigma_min, sigma_max)


def adaptive_loop(
    raw_step: AdmmRawStep,
    carry: AdaptiveCarry,
    y_raw: ProjectionInstance,
    omega: float | Float[Array, ""],
    n_iter: int,
) -> AdaptiveCarry:
    """Run ``n_iter`` Douglas-Rachford steps with residual-balanced ``sigma``.

    Args:
        raw_step: ADMM step returning ``(s_next, z, t)``.
        carry: Incoming solver carry.
        y_raw: Point being projected.
        omega: Relaxation parameter.
        n_iter: Number of steps to run.

    Returns:
        Carry after ``n_iter`` steps.
    """

    def body(state: AdaptiveCarry, _: None) -> tuple[AdaptiveCarry, None]:
        s_next, z, t = raw_step(state.s, y_raw, state.sigma, omega)
        period = jnp.int32(ADAPTIVE_EVERY)
        do_adapt = (state.iter_idx > 0) & (jnp.remainder(state.iter_idx + 1, period) == 0)

        def _adapt(_: None) -> Float[Array, ""]:
            primal_residual = _mean_norm(t - z)
            dual_residual = _mean_norm(z - state.z_prev) * state.sigma
            return adapt_sigma(state.sigma, primal_residual, dual_residual)

        def _keep(_: None) -> Float[Array, ""]:
            return state.sigma

        sigma_new = jax.lax.cond(do_adapt, _adapt, _keep, None)
        updated = AdaptiveCarry(
            s=s_next,
            sigma=sigma_new,
            z_prev=z,
            iter_idx=state.iter_idx + 1,
        )
        return updated, None

    scanned, _ = jax.lax.scan(body, carry, xs=None, length=n_iter)
    return scanned


def anderson_mix(
    x_hist: Float[Array, "m B d_lifted 1"],
    f_hist: Float[Array, "m B d_lifted 1"],
    n_filled: Integer[Array, ""],
    g_last: BatchedLifted,
    regularization: float = ANDERSON_REGULARIZATION,
    safeguard: float = ANDERSON_SAFEGUARD,
) -> BatchedLifted:
    """Type-II Anderson mixing of a batched fixed-point residual history.

    Solves a regularized least-squares problem per batch element on the
    residual differences, then forms the mixed operator output. The
    unaccelerated value ``g_last`` is returned when the history is too
    short or the mix is non-finite. History length 2 uses a closed-form
    secant step so the extra work is a handful of fused axpys.

    Args:
        x_hist: Past governing-sequence iterates, oldest first.
        f_hist: Past residuals ``T(s) - s``, oldest first.
        n_filled: Number of valid trailing history slots.
        g_last: Unaccelerated operator output ``T(s)``.
        regularization: Tikhonov term added to the Gram matrix.
        safeguard: Maximum allowed mixed-to-unaccelerated residual ratio.

    Returns:
        Mixed operator output, shaped like ``g_last``.
    """
    del safeguard
    memory = x_hist.shape[0]
    batch = g_last.shape[0]
    residuals = f_hist.reshape(memory, batch, -1)
    operator_hist = (x_hist + f_hist).reshape(memory, batch, -1)

    if memory == ANDERSON_MIN_HISTORY:
        d_residual = residuals[1] - residuals[0]
        denom = jnp.sum(d_residual * d_residual, axis=-1, keepdims=True) + regularization
        gamma = jnp.sum(d_residual * residuals[1], axis=-1, keepdims=True) / denom
        gamma = jnp.clip(gamma, -2.0, 2.0)
        mixed_flat = operator_hist[1] - gamma * (operator_hist[1] - operator_hist[0])
        mixed_residual = residuals[1] - gamma * d_residual
        last_sq = jnp.sum(residuals[1] * residuals[1], axis=-1)
        mixed_sq = jnp.sum(mixed_residual * mixed_residual, axis=-1)
        mixed_flat = jnp.where(
            mixed_sq[:, None] <= last_sq[:, None], mixed_flat, operator_hist[1]
        )
    else:
        d_residual = jnp.swapaxes(residuals[1:] - residuals[:-1], 0, 1)
        d_operator = jnp.swapaxes(operator_hist[1:] - operator_hist[:-1], 0, 1)
        gram = d_residual @ jnp.swapaxes(d_residual, 1, 2)
        eye = jnp.eye(memory - 1, dtype=gram.dtype)
        rhs = d_residual @ residuals[-1][:, :, None]
        gamma = jnp.linalg.solve(gram + regularization * eye, rhs)
        gamma = jnp.clip(gamma, -2.0, 2.0)
        correction = (jnp.swapaxes(d_operator, 1, 2) @ gamma).squeeze(-1)
        mixed_flat = operator_hist[-1] - correction
        mixed_residual = residuals[-1] - (jnp.swapaxes(d_residual, 1, 2) @ gamma).squeeze(
            -1
        )
        last_sq = jnp.sum(residuals[-1] * residuals[-1], axis=-1)
        mixed_sq = jnp.sum(mixed_residual * mixed_residual, axis=-1)
        mixed_flat = jnp.where(
            mixed_sq[:, None] <= last_sq[:, None], mixed_flat, operator_hist[-1]
        )

    mixed = mixed_flat.reshape(g_last.shape)
    mixed = jnp.where(jnp.isfinite(mixed), mixed, g_last)
    return jnp.where(n_filled >= ANDERSON_MIN_HISTORY, mixed, g_last)


def accelerated_loop(
    raw_step: AdmmRawStep,
    carry: AccelCarry,
    y_raw: ProjectionInstance,
    omega: float | Float[Array, ""],
    n_iter: int,
    *,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int = ANDERSON_MEMORY,
) -> AccelCarry:
    """Run ``n_iter`` accelerated Douglas-Rachford steps.

    Args:
        raw_step: ADMM step returning ``(s_next, z, t)``.
        carry: Incoming solver carry.
        y_raw: Point being projected.
        omega: Relaxation parameter.
        n_iter: Number of steps to run.
        use_anderson: Mix recent iterates with Type-II Anderson.
        use_adaptive_penalty: Residual-balance ``sigma``.
        anderson_memory: History length (must match ``carry``).

    Returns:
        Carry after ``n_iter`` steps.
    """
    del anderson_memory

    def body(state: AccelCarry, _: None) -> tuple[AccelCarry, None]:
        s_next, z, t = raw_step(state.s, y_raw, state.sigma, omega)
        g_last = s_next.x
        residual = g_last - state.s.x

        if use_anderson:
            if state.x_hist.shape[0] == ANDERSON_MIN_HISTORY:
                x_hist = jnp.stack([state.x_hist[1], state.s.x], axis=0)
                f_hist = jnp.stack([state.f_hist[1], residual], axis=0)
            else:
                x_hist = jnp.concatenate([state.x_hist[1:], state.s.x[None]], axis=0)
                f_hist = jnp.concatenate([state.f_hist[1:], residual[None]], axis=0)
            n_filled = jax.lax.min(
                state.n_filled + jnp.int32(1), jnp.int32(x_hist.shape[0])
            )
            mix_now = (n_filled >= ANDERSON_MIN_HISTORY) & (
                jnp.remainder(state.iter_idx + 1, jnp.int32(ANDERSON_EVERY)) == 0
            )
            residual_flat = residual.reshape(residual.shape[0], -1)
            mean_res = jnp.mean(jnp.linalg.norm(residual_flat, axis=-1))
            mix_now = mix_now & (mean_res > jnp.asarray(ANDERSON_STOP_RESIDUAL))

            def _do_mix(_: None) -> BatchedLifted:
                return anderson_mix(x_hist, f_hist, n_filled, g_last)

            def _skip_mix(_: None) -> BatchedLifted:
                return g_last

            if x_hist.shape[0] == ANDERSON_MIN_HISTORY:
                mixed = anderson_mix(x_hist, f_hist, n_filled, g_last)
                s_x = jnp.where(mix_now, mixed, g_last)
            else:
                s_x = jax.lax.cond(mix_now, _do_mix, _skip_mix, None)
            s_next = s_next.update(x=s_x)
        else:
            x_hist = state.x_hist
            f_hist = state.f_hist
            n_filled = state.n_filled

        if use_adaptive_penalty:
            period = jnp.int32(ADAPTIVE_EVERY)
            do_adapt = (state.iter_idx > 0) & (
                jnp.remainder(state.iter_idx + 1, period) == 0
            )

            def _adapt(_: None) -> Float[Array, ""]:
                primal_residual = _mean_norm(t - z)
                dual_residual = _mean_norm(z - state.z_prev) * state.sigma
                return adapt_sigma(state.sigma, primal_residual, dual_residual)

            def _keep(_: None) -> Float[Array, ""]:
                return state.sigma

            sigma_new = jax.lax.cond(do_adapt, _adapt, _keep, None)
            if use_anderson:
                changed = sigma_new != state.sigma
                x_hist = jnp.where(changed, jnp.zeros_like(x_hist), x_hist)
                f_hist = jnp.where(changed, jnp.zeros_like(f_hist), f_hist)
                n_filled = jax.lax.select(changed, jnp.int32(0), n_filled)
        else:
            sigma_new = state.sigma

        updated = AccelCarry(
            s=s_next,
            sigma=sigma_new,
            z_prev=z,
            x_hist=x_hist,
            f_hist=f_hist,
            n_filled=n_filled,
            iter_idx=state.iter_idx + 1,
        )
        return updated, None

    scanned, _ = jax.lax.scan(body, carry, xs=None, length=n_iter)
    return scanned


def identity_raw_step(
    sk: ProjectionInstance,
    y_raw: ProjectionInstance,
    sigma: float | Float[Array, ""],
    omega: float | Float[Array, ""],
) -> tuple[ProjectionInstance, BatchedLifted, BatchedLifted]:
    """No-op raw step used for the single-constraint closed form.

    Args:
        sk: Governing sequence.
        y_raw: Point being projected (unused).
        sigma: ADMM penalty (unused).
        omega: Relaxation parameter (unused).

    Returns:
        ``sk`` together with two copies of ``sk.x``.
    """
    del y_raw, sigma, omega
    return sk, sk.x, sk.x
