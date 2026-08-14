"""Compare pinet JAX, pinet Torch, and qpth on polyhedral projections.

Usage:

.. code-block:: console

    $ python -m src.benchmarks.torch.bench_project
    $ python -m src.benchmarks.torch.bench_project --dim 20 --repeats 10
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BATCH_SIZES = (1, 16, 64, 256, 1024)


@dataclass
class BenchResult:
    """Timing and accuracy for one solver at one batch size.

    Attributes:
        name: Solver name.
        batch_size: Number of problems.
        mean_ms: Mean wall time in milliseconds.
        std_ms: Standard deviation of wall time.
        throughput: Problems solved per second.
        max_diff_jax: Max abs difference vs JAX (None for the JAX row).
        max_cv: Max constraint violation.
        skipped: Skip reason, if the solver was not run.
    """

    name: str
    batch_size: int
    mean_ms: float | None
    std_ms: float | None
    throughput: float | None
    max_diff_jax: float | None
    max_cv: float | None
    skipped: str | None = None


def _make_problem(
    dim: int, n_eq: int, n_ineq: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a feasible polytope ``A y = b``, ``lb <= C y <= ub``.

    Args:
        dim: Primal dimension.
        n_eq: Number of equalities.
        n_ineq: Number of inequalities.
        seed: RNG seed.

    Returns:
        Tuple ``(A, b, C, lb, ub)`` with batch axis 1.
    """
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(1, n_eq, dim))
    c_mat = rng.normal(size=(1, n_ineq, dim))
    x_feas = rng.uniform(-1.0, 1.0, size=(1, dim, 1))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.5
    ub = c_mat @ x_feas + 0.5
    return a_mat, b, c_mat, lb, ub


def _polytope_cv(
    y: np.ndarray,
    a_mat: np.ndarray,
    b: np.ndarray,
    c_mat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> float:
    """Maximum equality/inequality violation of a batch of points.

    Args:
        y: Points, shape ``(B, dim)``.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        c_mat: Inequality matrix.
        lb: Inequality lower bound.
        ub: Inequality upper bound.

    Returns:
        Maximum residual over the batch.
    """
    eq = np.max(np.abs(a_mat[0] @ y.T - b[0, :, 0:1]), axis=0)
    cy = c_mat[0] @ y.T
    ineq = np.maximum(cy - ub[0, :, 0:1], lb[0, :, 0:1] - cy)
    ineq = np.maximum(ineq, 0.0)
    return float(max(np.max(eq), np.max(ineq)))


def _time_call(
    fn: Callable[[], object], warmup: int, repeats: int, sync: Callable[[object], None]
) -> tuple[float, float]:
    """Warm up, then time ``fn``.

    Args:
        fn: Zero-argument callable to time.
        warmup: Number of untimed calls.
        repeats: Number of timed calls.
        sync: Device-synchronization callback applied to each result.

    Returns:
        Mean and standard deviation in milliseconds.
    """
    for _ in range(warmup):
        sync(fn())
    samples: list[float] = []
    for _ in range(repeats):
        sync(None)
        start = time.perf_counter()
        out = fn()
        sync(out)
        samples.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(samples)
    return float(arr.mean()), float(arr.std())


def _run_jax(
    a_mat: np.ndarray,
    b: np.ndarray,
    c_mat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x: np.ndarray,
    n_iter: int,
    sigma: float,
    omega: float,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, float, float]:
    """Time the JAX projector.

    Args:
        a_mat: Equality matrix.
        b: Equality right-hand side.
        c_mat: Inequality matrix.
        lb: Inequality lower bound.
        ub: Inequality upper bound.
        x: Points to project, shape ``(B, dim)``.
        n_iter: ADMM iterations.
        sigma: ADMM stepsize.
        omega: Relaxation.
        warmup: Untimed calls.
        repeats: Timed calls.

    Returns:
        Tuple ``(y, mean_ms, std_ms)``.
    """
    import jax
    import jax.numpy as jnp

    from pinet import (
        AffineInequalityConstraint,
        EqualityConstraint,
        Project,
        ProjectionInstance,
    )

    jax.config.update("jax_enable_x64", True)
    proj = Project(
        eq_constraint=EqualityConstraint(jnp.array(a_mat), jnp.array(b), method="pinv"),
        ineq_constraint=AffineInequalityConstraint(
            jnp.array(c_mat), jnp.array(lb), jnp.array(ub)
        ),
    )
    x_j = jnp.array(x[:, :, None])

    def _call() -> jnp.ndarray:
        return proj.call(
            y_raw=ProjectionInstance(x=x_j), n_iter=n_iter, sigma=sigma, omega=omega
        )[0].x[..., 0]

    def _sync(result: object) -> None:
        if result is not None:
            jax.block_until_ready(result)

    mean_ms, std_ms = _time_call(_call, warmup, repeats, _sync)
    y = np.asarray(_call())
    return y, mean_ms, std_ms


def _run_torch(
    a_mat: np.ndarray,
    b: np.ndarray,
    c_mat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x: np.ndarray,
    n_iter: int,
    sigma: float,
    omega: float,
    warmup: int,
    repeats: int,
    device: str,
    compile_model: bool,
) -> tuple[np.ndarray, float, float]:
    """Time the Torch projector.

    Args:
        a_mat: Equality matrix.
        b: Equality right-hand side.
        c_mat: Inequality matrix.
        lb: Inequality lower bound.
        ub: Inequality upper bound.
        x: Points to project, shape ``(B, dim)``.
        n_iter: ADMM iterations.
        sigma: ADMM stepsize.
        omega: Relaxation.
        warmup: Untimed calls.
        repeats: Timed calls.
        device: Torch device string.
        compile_model: Whether to request ``torch.compile``.

    Returns:
        Tuple ``(y, mean_ms, std_ms)``.
    """
    import torch

    from pinet.torch import AffineInequalityConstraint, EqualityConstraint, Project

    proj = Project(
        EqualityConstraint(a_mat, b, method="pinv"),
        AffineInequalityConstraint(c_mat, lb, ub),
        n_iter=n_iter,
        compile=compile_model,
        compile_mode="reduce-overhead" if device == "cuda" else "default",
    ).to(device)
    x_t = torch.tensor(x, dtype=torch.float64, device=device)

    def _sync(result: object) -> None:
        del result
        if device == "cuda":
            torch.cuda.synchronize()

    def _call() -> torch.Tensor:
        with torch.no_grad():
            return proj(x_t, n_iter=n_iter, sigma=sigma, omega=omega)

    mean_ms, std_ms = _time_call(_call, warmup, repeats, _sync)
    y = _call().detach().cpu().numpy()
    return y, mean_ms, std_ms


def _run_qpth(
    a_mat: np.ndarray,
    b: np.ndarray,
    c_mat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x: np.ndarray,
    warmup: int,
    repeats: int,
    device: str,
) -> tuple[np.ndarray, float, float]:
    """Time qpth on the equivalent QP.

    Args:
        a_mat: Equality matrix.
        b: Equality right-hand side.
        c_mat: Inequality matrix.
        lb: Inequality lower bound.
        ub: Inequality upper bound.
        x: Points to project, shape ``(B, dim)``.
        warmup: Untimed calls.
        repeats: Timed calls.
        device: Torch device string.

    Returns:
        Tuple ``(y, mean_ms, std_ms)``.
    """
    import torch

    qp_mod = importlib.import_module("qpth.qp")
    qp_function = qp_mod.QPFunction

    batch, dim = x.shape
    q_mat = torch.eye(dim, dtype=torch.float64, device=device).expand(batch, dim, dim)
    p_vec = -torch.tensor(x, dtype=torch.float64, device=device)
    g_mat = torch.tensor(
        np.concatenate([c_mat[0], -c_mat[0]], axis=0),
        dtype=torch.float64,
        device=device,
    ).expand(batch, 2 * c_mat.shape[1], dim)
    h_vec = torch.tensor(
        np.concatenate([ub[0, :, 0], -lb[0, :, 0]], axis=0),
        dtype=torch.float64,
        device=device,
    ).expand(batch, 2 * c_mat.shape[1])
    a_t = torch.tensor(a_mat[0], dtype=torch.float64, device=device).expand(
        batch, a_mat.shape[1], dim
    )
    b_t = torch.tensor(b[0, :, 0], dtype=torch.float64, device=device).expand(
        batch, a_mat.shape[1]
    )
    qp = qp_function(verbose=False)

    def _sync(result: object) -> None:
        del result
        if device == "cuda":
            torch.cuda.synchronize()

    def _call() -> torch.Tensor:
        with torch.no_grad():
            return qp(q_mat, p_vec, g_mat, h_vec, a_t, b_t)

    mean_ms, std_ms = _time_call(_call, warmup, repeats, _sync)
    y = _call().detach().cpu().numpy()
    return y, mean_ms, std_ms


def run_benchmark(
    dim: int = 100,
    n_eq: int = 50,
    n_ineq: int = 50,
    n_iter: int = 50,
    sigma: float = 1.0,
    omega: float = 1.7,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = 5,
    repeats: int = 20,
    seed: int = 0,
    device: str | None = None,
) -> list[BenchResult]:
    """Run the three-way projection benchmark.

    Args:
        dim: Primal dimension.
        n_eq: Number of equalities.
        n_ineq: Number of inequalities.
        n_iter: ADMM iterations for pinet.
        sigma: ADMM stepsize.
        omega: Relaxation.
        batch_sizes: Batch sizes to sweep.
        warmup: Untimed calls per solver.
        repeats: Timed calls per solver.
        seed: Problem RNG seed.
        device: Torch device; inferred from CUDA availability when omitted.

    Returns:
        One result row per solver and batch size.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    a_mat, b, c_mat, lb, ub = _make_problem(dim, n_eq, n_ineq, seed)
    rng = np.random.default_rng(seed + 1)
    results: list[BenchResult] = []
    qpth_error: str | None = None
    if importlib.util.find_spec("qpth") is None:
        qpth_error = "qpth is not installed (pip install qpth --no-deps)"

    for batch_size in batch_sizes:
        x = rng.uniform(-2.0, 2.0, size=(batch_size, dim))
        y_jax, jax_mean, jax_std = _run_jax(
            a_mat, b, c_mat, lb, ub, x, n_iter, sigma, omega, warmup, repeats
        )
        results.append(
            BenchResult(
                name="pinet-jax",
                batch_size=batch_size,
                mean_ms=jax_mean,
                std_ms=jax_std,
                throughput=1000.0 * batch_size / jax_mean,
                max_diff_jax=0.0,
                max_cv=_polytope_cv(y_jax, a_mat, b, c_mat, lb, ub),
            )
        )
        y_torch, torch_mean, torch_std = _run_torch(
            a_mat,
            b,
            c_mat,
            lb,
            ub,
            x,
            n_iter,
            sigma,
            omega,
            warmup,
            repeats,
            device,
            compile_model=device == "cuda",
        )
        results.append(
            BenchResult(
                name="pinet-torch",
                batch_size=batch_size,
                mean_ms=torch_mean,
                std_ms=torch_std,
                throughput=1000.0 * batch_size / torch_mean,
                max_diff_jax=float(np.max(np.abs(y_torch - y_jax))),
                max_cv=_polytope_cv(y_torch, a_mat, b, c_mat, lb, ub),
            )
        )
        if qpth_error is not None:
            results.append(
                BenchResult(
                    name="qpth",
                    batch_size=batch_size,
                    mean_ms=None,
                    std_ms=None,
                    throughput=None,
                    max_diff_jax=None,
                    max_cv=None,
                    skipped=qpth_error,
                )
            )
            continue
        try:
            y_qpth, q_mean, q_std = _run_qpth(
                a_mat, b, c_mat, lb, ub, x, warmup, repeats, device
            )
            results.append(
                BenchResult(
                    name="qpth",
                    batch_size=batch_size,
                    mean_ms=q_mean,
                    std_ms=q_std,
                    throughput=1000.0 * batch_size / q_mean,
                    max_diff_jax=float(np.max(np.abs(y_qpth - y_jax))),
                    max_cv=_polytope_cv(y_qpth, a_mat, b, c_mat, lb, ub),
                )
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            results.append(
                BenchResult(
                    name="qpth",
                    batch_size=batch_size,
                    mean_ms=None,
                    std_ms=None,
                    throughput=None,
                    max_diff_jax=None,
                    max_cv=None,
                    skipped=str(exc),
                )
            )
    return results


def _format_table(results: list[BenchResult], device: str) -> str:
    """Render a text table of benchmark rows.

    Args:
        results: Benchmark rows.
        device: Device used for Torch/qpth.

    Returns:
        Printable table.
    """
    header = (
        f"{'solver':<14} {'B':>6} {'mean_ms':>12} {'std_ms':>10} "
        f"{'probs/s':>12} {'||Δjax||':>12} {'max_cv':>12}"
    )
    lines = [f"device={device}", header, "-" * len(header)]
    for row in results:
        if row.skipped:
            lines.append(f"{row.name:<14} {row.batch_size:>6}   skipped: {row.skipped}")
            continue
        assert row.mean_ms is not None
        assert row.std_ms is not None
        assert row.throughput is not None
        assert row.max_cv is not None
        diff = "n/a" if row.max_diff_jax is None else f"{row.max_diff_jax:.3e}"
        lines.append(
            f"{row.name:<14} {row.batch_size:>6} {row.mean_ms:>12.3f} "
            f"{row.std_ms:>10.3f} {row.throughput:>12.1f} {diff:>12} "
            f"{row.max_cv:>12.3e}"
        )
    return "\n".join(lines)


def main() -> None:
    """Parse CLI arguments and print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--n-eq", type=int, default=50)
    parser.add_argument("--n-ineq", type=int, default=50)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(BATCH_SIZES),
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = run_benchmark(
        dim=args.dim,
        n_eq=args.n_eq,
        n_ineq=args.n_ineq,
        n_iter=args.n_iter,
        batch_sizes=tuple(args.batch_sizes),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    print(_format_table(results, device))


if __name__ == "__main__":
    main()
