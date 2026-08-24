"""Sweep residual balancing of ``sigma`` against the fused original loop.

Reports iteration count, wall-clock time, and the adapted penalty for a
grid of initial ``sigma`` values, several constraint families, ill-conditioned
equalities, and tight vs loose inequalities.

The fused original loop (no residual balancing) is the baseline. Adaptive
penalty is most useful when the initial ``sigma`` is far from a well-scaled
value near 1; at that default it typically matches the original iteration
count.

.. code-block:: console

    $ uv run python src/benchmarks/compare_adaptive_penalty.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)
from pinet.solver.acceleration import AdaptiveCarry, adaptive_loop, init_adaptive_carry

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ProblemSpec:
    """Random feasible projection problem used by the adaptive sweep."""

    name: str
    batch_size: int
    dim: int
    n_eq: int
    n_ineq: int
    seed: int = 0
    slack: float = 0.25
    row_scale: float = 1.0
    family: str = "eq_ineq"


REPEATS = 3
CHECK_EVERY = 10
TOL = 1e-4
MAX_ITER = 500
OMEGA = 1.7

SIGMA_GRID: tuple[float, ...] = (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

SIZES: tuple[ProblemSpec, ...] = (
    ProblemSpec("small", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=0),
    ProblemSpec("medium", batch_size=32, dim=80, n_eq=30, n_ineq=40, seed=1),
    ProblemSpec("large", batch_size=64, dim=120, n_eq=40, n_ineq=50, seed=2),
)

SEED_SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SEED_SWEEP_SIGMAS: tuple[float, ...] = (0.05, 1.0, 20.0)
SEED_SWEEP_TEMPLATE = ProblemSpec(
    "seed", batch_size=16, dim=48, n_eq=16, n_ineq=20, seed=0
)

FAMILIES: tuple[ProblemSpec, ...] = (
    ProblemSpec(
        "eq_ineq", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=0, family="eq_ineq"
    ),
    ProblemSpec(
        "eq_box", batch_size=16, dim=32, n_eq=12, n_ineq=0, seed=1, family="eq_box"
    ),
    ProblemSpec(
        "ineq_only", batch_size=16, dim=32, n_eq=0, n_ineq=24, seed=2, family="ineq_only"
    ),
)
FAMILY_SIGMAS: tuple[float, ...] = (0.05, 1.0, 20.0)

CONDITIONING: tuple[ProblemSpec, ...] = (
    ProblemSpec("well", batch_size=16, dim=40, n_eq=16, n_ineq=16, seed=4, row_scale=1.0),
    ProblemSpec("ill", batch_size=16, dim=40, n_eq=16, n_ineq=16, seed=4, row_scale=1e2),
)
TIGHTNESS: tuple[ProblemSpec, ...] = (
    ProblemSpec("tight", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=5, slack=0.05),
    ProblemSpec("mid", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=5, slack=0.25),
    ProblemSpec("loose", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=5, slack=1.0),
)


def _make_problem(spec: ProblemSpec) -> tuple[Project, ProjectionInstance]:
    """Build a feasible random instance of the requested constraint family."""
    key = jax.random.PRNGKey(spec.seed)
    ka_mat, kfeas, kc_mat, kx, kscale = jax.random.split(key, 5)
    x_feas = jax.random.normal(kfeas, (1, spec.dim, 1))
    xinfeas = jax.random.normal(kx, (spec.batch_size, spec.dim, 1))

    eq: EqualityConstraint | None = None
    ineq: AffineInequalityConstraint | None = None
    box: BoxConstraint | None = None

    if spec.n_eq > 0:
        a_mat = jax.random.normal(ka_mat, (1, spec.n_eq, spec.dim))
        if spec.row_scale != 1.0:
            scales = spec.row_scale ** jax.random.uniform(
                kscale, (spec.n_eq,), minval=-1.0, maxval=1.0
            )
            a_mat = a_mat * scales[None, :, None]
        b = a_mat @ x_feas
        eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)

    if spec.family == "eq_box":
        width = 1.5
        lb = x_feas - width
        ub = x_feas + width
        box = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    elif spec.n_ineq > 0:
        c_mat = jax.random.normal(kc_mat, (1, spec.n_ineq, spec.dim))
        lb = c_mat @ x_feas - spec.slack
        ub = c_mat @ x_feas + spec.slack
        ineq = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    layer = Project(eq_constraint=eq, ineq_constraint=ineq, box_constraint=box)
    return layer, ProjectionInstance(x=xinfeas)


def _finalize(
    layer: Project, sk: ProjectionInstance, y_raw: ProjectionInstance
) -> ProjectionInstance:
    """Unscale the equality-block projection back to the original variables.

    Args:
        layer: Projection layer.
        sk: Governing sequence.
        y_raw: Original projection request.

    Returns:
        Projected point in the original coordinates.
    """
    primal_dim = y_raw.x.shape[1]
    y = layer.step_final(sk).x[:, :primal_dim, :]
    return y_raw.update(x=y * layer.d_c[:, :primal_dim, :])


def _make_solvers(
    layer: Project,
) -> tuple[
    Callable[
        [ProjectionInstance, jax.Array],
        tuple[ProjectionInstance, jax.Array, jax.Array],
    ],
    Callable[
        [ProjectionInstance, jax.Array],
        tuple[ProjectionInstance, jax.Array, jax.Array, jax.Array],
    ],
]:
    """Build fused original and adaptive solvers with runtime ``sigma``.

    Args:
        layer: Projection layer.

    Returns:
        ``(solve_original, solve_adaptive)``. The adaptive solver also
        returns the final penalty.
    """
    raw_step = layer.raw_step
    step_iteration = layer.step_iteration
    initialize_fn = layer.initialize

    @jax.jit
    def check(inp: ProjectionInstance) -> jax.Array:
        return jnp.max(layer.cv(inp)) < TOL

    @jax.jit
    def solve_original(
        y_raw: ProjectionInstance, sigma: jax.Array
    ) -> tuple[ProjectionInstance, jax.Array, jax.Array]:
        s0 = initialize_fn(y_raw)

        def cond(
            state: tuple[ProjectionInstance, ProjectionInstance, jax.Array, jax.Array],
        ) -> jax.Array:
            _, _, it, done = state
            return jnp.logical_and(jnp.logical_not(done), it < MAX_ITER)

        def body(
            state: tuple[ProjectionInstance, ProjectionInstance, jax.Array, jax.Array],
        ) -> tuple[ProjectionInstance, ProjectionInstance, jax.Array, jax.Array]:
            _, s_in, it, _ = state
            s_out, _ = jax.lax.scan(
                lambda s_prev, _: (step_iteration(s_prev, y_raw, sigma, OMEGA), None),
                s_in,
                None,
                length=CHECK_EVERY,
            )
            xproj = _finalize(layer, s_out, y_raw)
            return xproj, s_out, it + jnp.int32(CHECK_EVERY), check(xproj)

        xproj, _, it, done = jax.lax.while_loop(
            cond, body, (y_raw, s0, jnp.int32(0), jnp.array(False))
        )
        return xproj, done, it

    @jax.jit
    def solve_adaptive(
        y_raw: ProjectionInstance, sigma: jax.Array
    ) -> tuple[ProjectionInstance, jax.Array, jax.Array, jax.Array]:
        s0 = initialize_fn(y_raw)
        carry0 = init_adaptive_carry(s0, sigma)

        def cond(
            state: tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array],
        ) -> jax.Array:
            _, _, it, done = state
            return jnp.logical_and(jnp.logical_not(done), it < MAX_ITER)

        def body(
            state: tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array],
        ) -> tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array]:
            _, carry_in, it, _ = state
            carry_out = adaptive_loop(raw_step, carry_in, y_raw, OMEGA, CHECK_EVERY)
            xproj = _finalize(layer, carry_out.s, y_raw)
            return xproj, carry_out, it + jnp.int32(CHECK_EVERY), check(xproj)

        xproj, carry, it, done = jax.lax.while_loop(
            cond, body, (y_raw, carry0, jnp.int32(0), jnp.array(False))
        )
        return xproj, done, it, carry.sigma

    return solve_original, solve_adaptive


def _max_cv(layer: Project, y: ProjectionInstance) -> float:
    """Return the batch-max constraint violation.

    Args:
        layer: Projection layer.
        y: Candidate projection.

    Returns:
        Maximum constraint violation over the batch.
    """
    return float(jnp.max(layer.cv(y)))


def _mean_ms(times: list[float]) -> float:
    """Convert a list of durations in seconds to a mean in milliseconds.

    Args:
        times: Wall-clock durations in seconds.

    Returns:
        Mean duration in milliseconds.
    """
    return (sum(times) / len(times)) * 1e3


def _compare_one(
    spec: ProblemSpec,
    sigma: float,
    layer: Project,
    y_raw: ProjectionInstance,
    solve_original: Callable[
        [ProjectionInstance, jax.Array],
        tuple[ProjectionInstance, jax.Array, jax.Array],
    ],
    solve_adaptive: Callable[
        [ProjectionInstance, jax.Array],
        tuple[ProjectionInstance, jax.Array, jax.Array, jax.Array],
    ],
) -> dict[str, float | int | str | bool]:
    """Compare fused original vs adaptive penalty on one problem.

    Args:
        spec: Problem description (name only is recorded).
        sigma: Initial ADMM penalty.
        layer: Projection layer (used for constraint-violation checks).
        y_raw: Infeasible batch.
        solve_original: Fused unaccelerated solver.
        solve_adaptive: Fused residual-balancing solver.

    Returns:
        Metrics for the original and adaptive solves.
    """
    sigma_arr = jnp.asarray(sigma)
    y0, flag0, it0 = solve_original(y_raw, sigma_arr)
    jax.block_until_ready(y0.x)
    times0: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        y0, flag0, it0 = solve_original(y_raw, sigma_arr)
        jax.block_until_ready(y0.x)
        times0.append(time.perf_counter() - start)

    y1, flag1, it1, sigma_final = solve_adaptive(y_raw, sigma_arr)
    jax.block_until_ready(y1.x)
    times1: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        y1, flag1, it1, sigma_final = solve_adaptive(y_raw, sigma_arr)
        jax.block_until_ready(y1.x)
        times1.append(time.perf_counter() - start)

    ms0 = _mean_ms(times0)
    ms1 = _mean_ms(times1)
    speedup = ms0 / ms1 if ms1 > 0 else float("inf")
    return {
        "name": spec.name,
        "sigma0": sigma,
        "orig_iters": int(it0),
        "ad_iters": int(it1),
        "orig_ms": ms0,
        "ad_ms": ms1,
        "speedup": speedup,
        "sigma_final": float(sigma_final),
        "orig_cv": _max_cv(layer, y0),
        "ad_cv": _max_cv(layer, y1),
        "orig_ok": bool(flag0),
        "ad_ok": bool(flag1),
    }


def _print_row(row: dict[str, float | int | str | bool], name_width: int = 10) -> None:
    """Print one comparison row.

    Args:
        row: Metrics from ``_compare_one``.
        name_width: Width of the leading name column.
    """
    ok = "yes" if row["orig_ok"] and row["ad_ok"] else "no"
    print(
        f"{row['name']!s:<{name_width}} {float(row['sigma0']):>8.3g} "
        f"{int(row['orig_iters']):>8} {int(row['ad_iters']):>8} "
        f"{float(row['orig_ms']):>9.2f} {float(row['ad_ms']):>9.2f} "
        f"{float(row['speedup']):>8.2f}x {float(row['sigma_final']):>10.4g} "
        f"{float(row['ad_cv']):>10.2e} {ok:>4}"
    )


def _print_header(title: str, name_width: int = 10) -> None:
    """Print a section title and column header.

    Args:
        title: Section heading.
        name_width: Width of the leading name column.
    """
    header = (
        f"{'case':<{name_width}} {'sigma0':>8} {'orig_it':>8} {'ad_it':>8} "
        f"{'orig_ms':>9} {'ad_ms':>9} {'speedup':>9} {'sigma*':>10} "
        f"{'ad_cv':>10} {'ok':>4}"
    )
    print()
    print(title)
    print(header)
    print("-" * len(header))


def _sigma_trajectory(
    spec: ProblemSpec,
    sigma0: float,
    n_iter: int = 80,
    period: int = 5,
) -> None:
    """Print the adapted penalty every ``period`` steps.

    Args:
        spec: Problem description.
        sigma0: Initial penalty.
        n_iter: Total Douglas-Rachford steps.
        period: Print period.
    """
    layer, y_raw = _make_problem(spec)
    s0 = layer.initialize(y_raw)
    carry = init_adaptive_carry(s0, sigma0)

    @jax.jit
    def _chunk(state: AdaptiveCarry) -> AdaptiveCarry:
        return adaptive_loop(layer.raw_step, state, y_raw, OMEGA, period)

    print()
    print(f"Sigma trajectory ({spec.name}, sigma0={sigma0:g}, every {period} steps)")
    print(f"{'iter':>6} {'sigma':>12} {'cv':>12}")
    print("-" * 32)
    for step in range(0, n_iter + 1, period):
        if step > 0:
            carry = _chunk(carry)
        xproj = _finalize(layer, carry.s, y_raw)
        cv = _max_cv(layer, xproj)
        print(f"{step:6d} {float(carry.sigma):12.5g} {cv:12.3e}")


def _eval_spec(
    spec: ProblemSpec,
    sigmas: tuple[float, ...],
    name_width: int = 10,
) -> list[dict[str, float | int | str | bool]]:
    """Build one problem, compile solvers once, and sweep ``sigmas``.

    Args:
        spec: Problem description.
        sigmas: Initial penalties to try.
        name_width: Width of the printed name column.

    Returns:
        One metrics dict per sigma.
    """
    layer, y_raw = _make_problem(spec)
    solve_original, solve_adaptive = _make_solvers(layer)
    rows: list[dict[str, float | int | str | bool]] = []
    for sigma in sigmas:
        row = _compare_one(spec, sigma, layer, y_raw, solve_original, solve_adaptive)
        _print_row(row, name_width=name_width)
        rows.append(row)
    return rows


def run() -> None:
    """Print adaptive-penalty comparison tables."""
    _print_header("1. Initial-sigma grid (eq + affine inequality)")
    for spec in SIZES:
        _eval_spec(spec, SIGMA_GRID)

    _print_header("2. Seeds (eq+ineq, dim=48)", name_width=12)
    seed_rows: list[dict[str, float | int | str | bool]] = []
    for seed in SEED_SWEEP_SEEDS:
        spec = ProblemSpec(
            name=f"s{seed}",
            batch_size=SEED_SWEEP_TEMPLATE.batch_size,
            dim=SEED_SWEEP_TEMPLATE.dim,
            n_eq=SEED_SWEEP_TEMPLATE.n_eq,
            n_ineq=SEED_SWEEP_TEMPLATE.n_ineq,
            seed=seed,
        )
        seed_rows.extend(_eval_spec(spec, SEED_SWEEP_SIGMAS, name_width=12))
    for sigma in SEED_SWEEP_SIGMAS:
        subset = [row for row in seed_rows if float(row["sigma0"]) == sigma]
        speedups = [float(row["speedup"]) for row in subset]
        iter_ratios = [
            int(row["ad_iters"]) / int(row["orig_iters"])
            for row in subset
            if int(row["orig_iters"]) > 0
        ]
        geo = float(jnp.exp(jnp.mean(jnp.log(jnp.asarray(speedups)))))
        mean_ir = float(jnp.mean(jnp.asarray(iter_ratios)))
        print(
            f"{'geo-mean':<12} {sigma:>8.3g} "
            f"{'':>8} {mean_ir:8.2f} {'':>9} {'':>9} {geo:8.2f}x"
        )

    _print_header("3. Constraint families", name_width=12)
    for spec in FAMILIES:
        _eval_spec(spec, FAMILY_SIGMAS, name_width=12)

    _print_header("4. Equality row scaling (conditioning)")
    for spec in CONDITIONING:
        _eval_spec(spec, FAMILY_SIGMAS)

    _print_header("5. Inequality tightness")
    for spec in TIGHTNESS:
        _eval_spec(spec, FAMILY_SIGMAS)

    _sigma_trajectory(SIZES[0], sigma0=0.05, n_iter=80, period=5)
    _sigma_trajectory(SIZES[0], sigma0=20.0, n_iter=80, period=5)
    _sigma_trajectory(SIZES[0], sigma0=1.0, n_iter=40, period=5)


if __name__ == "__main__":
    run()
