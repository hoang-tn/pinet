"""Anderson acceleration and residual-balancing helpers for the ADMM solver."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from pinet._typing import ScalarLike
from pinet.constants import Constants
from pinet.dataclasses import ProjectionInstance

AdmmRawStep = Callable[
    [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
    tuple[ProjectionInstance, jax.Array, jax.Array],
]

ANDERSON_MEMORY = Constants.PROJECTION_ANDERSON_MEMORY
ANDERSON_REGULARIZATION = Constants.PROJECTION_ANDERSON_REGULARIZATION
ANDERSON_SAFEGUARD = Constants.PROJECTION_ANDERSON_SAFEGUARD
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
    z_prev: Float[Array, "B d_lifted 1"]
    x_hist: Float[Array, "m B d_lifted 1"]
    f_hist: Float[Array, "m B d_lifted 1"]
    n_filled: Integer[Array, ""]
    iter_idx: Integer[Array, ""]


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


def _mean_norm(residual: Float[Array, "B d_lifted 1"]) -> Float[Array, ""]:
    """Mean Euclidean residual over the batch."""
    per_sample = jnp.linalg.norm(residual.reshape(residual.shape[0], -1), axis=-1)
    return jnp.mean(per_sample)


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
    sigma_new = jnp.where(
        too_primal, increased, jnp.where(too_dual, decreased, sigma)
    )
    return jnp.clip(sigma_new, sigma_min, sigma_max)


def anderson_mix(
    x_hist: Float[Array, "m B d_lifted 1"],
    f_hist: Float[Array, "m B d_lifted 1"],
    n_filled: Integer[Array, ""],
    g_last: Float[Array, "B d_lifted 1"],
    regularization: float = ANDERSON_REGULARIZATION,
    safeguard: float = ANDERSON_SAFEGUARD,
) -> Float[Array, "B d_lifted 1"]:
    """Type-II Anderson mixing of a batched fixed-point residual history.

    Solves a regularized least-squares problem per batch element on the
    residual differences, then forms the mixed operator output. The
    unaccelerated value ``g_last`` is returned when the history is too
    short, the mix is non-finite, or the mixed residual exceeds
    ``safeguard`` times the unaccelerated residual.

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
    memory = x_hist.shape[0]
    batch = g_last.shape[0]
    valid = jnp.arange(memory) >= (memory - n_filled)
    valid_diff = valid[1:] & valid[:-1]

    residuals = f_hist.reshape(memory, batch, -1)
    operator_hist = (x_hist + f_hist).reshape(memory, batch, -1)
    d_residual = (residuals[1:] - residuals[:-1]) * valid_diff[:, None, None]
    d_operator = operator_hist[1:] - operator_hist[:-1]
    residual_last = residuals[-1]

    gram = jnp.einsum("kbi,lbi->bkl", d_residual, d_residual)
    rhs = jnp.einsum("kbi,bi->bk", d_residual, residual_last)
    fill_diag = jnp.ones_like(valid_diff, dtype=gram.dtype)
    diag_reg = jnp.where(valid_diff, regularization, fill_diag)
    gram = gram + jnp.diag(diag_reg)
    gamma = jnp.linalg.solve(gram, rhs)

    mixed_flat = operator_hist[-1] - jnp.einsum("kbi,bk->bi", d_operator, gamma)
    mixed = mixed_flat.reshape(g_last.shape)
    mixed_residual = residual_last - jnp.einsum("kbi,bk->bi", d_residual, gamma)
    mixed_norm = jnp.linalg.norm(mixed_residual, axis=-1)
    last_norm = jnp.linalg.norm(residual_last, axis=-1)
    accept = (mixed_norm <= safeguard * last_norm) & jnp.isfinite(mixed_norm)
    guarded = jnp.where(accept[:, None, None], mixed, g_last)
    return jnp.where(n_filled >= 2, guarded, g_last)


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
        primal_residual = _mean_norm(t - z)
        dual_residual = _mean_norm(z - state.z_prev)
        g_last = s_next.x
        residual = g_last - state.s.x

        if use_anderson:
            x_hist = jnp.concatenate([state.x_hist[1:], state.s.x[None]], axis=0)
            f_hist = jnp.concatenate([state.f_hist[1:], residual[None]], axis=0)
            n_filled = jnp.minimum(state.n_filled + 1, x_hist.shape[0]).astype(jnp.int32)
            s_next = s_next.update(x=anderson_mix(x_hist, f_hist, n_filled, g_last))
        else:
            x_hist = state.x_hist
            f_hist = state.f_hist
            n_filled = state.n_filled

        if use_adaptive_penalty:
            period = jnp.int32(ADAPTIVE_EVERY)
            do_adapt = (state.iter_idx > 0) & (
                jnp.remainder(state.iter_idx + 1, period) == 0
            )
            sigma_candidate = adapt_sigma(
                state.sigma, primal_residual, dual_residual
            )
            sigma_new = jnp.where(do_adapt, sigma_candidate, state.sigma)
            if use_anderson:
                changed = sigma_new != state.sigma
                x_hist = jnp.where(changed, jnp.zeros_like(x_hist), x_hist)
                f_hist = jnp.where(changed, jnp.zeros_like(f_hist), f_hist)
                n_filled = jnp.where(changed, jnp.int32(0), n_filled)
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
) -> tuple[ProjectionInstance, jax.Array, jax.Array]:
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
