"""Compare original vs accelerated Douglas-Rachford on random projections.

Reports wall-clock time and iteration count of ``call_and_check`` for the
original ADMM map against Type-II Anderson acceleration, residual
balancing, and both together. ``host_loop`` is the pre-fusion Python
``while`` that synchronized with the host on every feasibility check;
``original`` is the same unaccelerated map inside a device-side
``while_loop``.

.. code-block:: console

    $ uv run python src/benchmarks/compare_acceleration.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class BenchCase:
    """One projection problem used by the acceleration comparison."""

    name: str
    batch_size: int
    dim: int
    n_eq: int
    n_ineq: int
    seed: int = 0
    sigma: float = 1.0


CASES: tuple[BenchCase, ...] = (
    BenchCase("small", batch_size=16, dim=32, n_eq=12, n_ineq=16, seed=0),
    BenchCase("medium", batch_size=32, dim=80, n_eq=30, n_ineq=40, seed=1),
    BenchCase("large", batch_size=64, dim=120, n_eq=40, n_ineq=50, seed=2),
    BenchCase("xlarge", batch_size=32, dim=400, n_eq=120, n_ineq=120, seed=3),
    BenchCase(
        "bad_sigma",
        batch_size=16,
        dim=32,
        n_eq=12,
        n_ineq=16,
        seed=0,
        sigma=0.05,
    ),
)

VARIANTS: tuple[tuple[str, bool, bool], ...] = (
    ("original", False, False),
    ("anderson", True, False),
    ("adaptive_penalty", False, True),
    ("anderson+adaptive", True, True),
)

REPEATS = 5
CHECK_EVERY = 10
TOL = 1e-4
MAX_ITER = 500
OMEGA = 1.7


def _make_problem(case: BenchCase) -> tuple[Project, ProjectionInstance]:
    """Build a feasible random equality-plus-inequality instance."""
    key = jax.random.PRNGKey(case.seed)
    ka_mat, kfeas, kc_mat, kx = jax.random.split(key, 4)
    a_mat = jax.random.normal(ka_mat, (1, case.n_eq, case.dim))
    x_feas = jax.random.normal(kfeas, (1, case.dim, 1))
    b = a_mat @ x_feas
    c_mat = jax.random.normal(kc_mat, (1, case.n_ineq, case.dim))
    slack = 0.25
    lb = c_mat @ x_feas - slack
    ub = c_mat @ x_feas + slack
    xinfeas = jax.random.normal(kx, (case.batch_size, case.dim, 1))
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)
    ineq = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)
    layer = Project(eq_constraint=eq, ineq_constraint=ineq)
    return layer, ProjectionInstance(x=xinfeas)


def _host_loop_solver(
    layer: Project,
    sigma: float,
) -> Callable[[ProjectionInstance], tuple[ProjectionInstance, jax.Array, int]]:
    """Rebuild the pre-fusion Python while-loop around ``Project.call``.

    Args:
        layer: Projection layer whose unaccelerated ``call`` is stepped.
        sigma: ADMM penalty.

    Returns:
        Solve-to-tolerance callable matching ``call_and_check``.
    """
    check_every = CHECK_EVERY
    max_iter = MAX_ITER
    tol = TOL

    @jax.jit
    def _check(inp: ProjectionInstance) -> jax.Array:
        return jnp.max(layer.cv(inp)) < tol

    def project_and_check(
        y_raw: ProjectionInstance,
    ) -> tuple[ProjectionInstance, jax.Array, int]:
        iter_exec = 0
        terminate = False
        y0 = layer.initialize(y_raw)
        xproj = y_raw
        while not (terminate or iter_exec >= max_iter):
            xproj, y0 = layer.call(
                s0=y0,
                y_raw=y_raw,
                sigma=sigma,
                omega=OMEGA,
                n_iter=check_every,
            )
            iter_exec += check_every
            terminate = bool(_check(xproj))
        return xproj, jnp.array(terminate), iter_exec

    return project_and_check


def _max_cv(layer: Project, y: ProjectionInstance) -> float:
    """Return the batch-max constraint violation.

    Args:
        layer: Projection layer.
        y: Candidate projection.

    Returns:
        Maximum constraint violation over the batch.
    """
    return float(jnp.max(layer.cv(y)))


def run() -> None:
    """Print a comparison table for every case and solver variant."""
    header = (
        f"{'case':<8} {'variant':<20} {'iters':>7} {'time_ms':>10} "
        f"{'speedup':>8} {'cv':>10} {'ok':>4}"
    )
    print(header)
    print("-" * len(header))
    for case in CASES:
        layer, y_raw = _make_problem(case)
        baseline_ms: float | None = None
        host_solver = _host_loop_solver(layer, case.sigma)
        solvers: list[
            tuple[
                str,
                Callable[
                    [ProjectionInstance],
                    tuple[ProjectionInstance, jax.Array, jax.Array | int],
                ],
            ]
        ] = [("host_loop", host_solver)]
        for name, use_anderson, use_adaptive in VARIANTS:
            solvers.append(
                (
                    name,
                    layer.call_and_check(
                        sigma=case.sigma,
                        omega=OMEGA,
                        check_every=CHECK_EVERY,
                        tol=TOL,
                        max_iter=MAX_ITER,
                        reduction="max",
                        use_anderson=use_anderson,
                        use_adaptive_penalty=use_adaptive,
                    ),
                )
            )
        for name, solver in solvers:
            y, flag, iters = solver(y_raw)
            jax.block_until_ready(y.x)
            times: list[float] = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                y, flag, iters = solver(y_raw)
                jax.block_until_ready(y.x)
                times.append(time.perf_counter() - start)
            mean_ms = (sum(times) / len(times)) * 1e3
            if baseline_ms is None:
                baseline_ms = mean_ms
            speedup = baseline_ms / mean_ms if mean_ms > 0 else float("inf")
            cv = _max_cv(layer, y)
            ok = "yes" if bool(flag) else "no"
            print(
                f"{case.name:<8} {name:<20} {int(iters):>7} {mean_ms:10.2f} "
                f"{speedup:8.2f}x {cv:10.2e} {ok:>4}"
            )


if __name__ == "__main__":
    run()
