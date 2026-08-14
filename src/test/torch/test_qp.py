"""Accuracy and gradient checks for the fast batched PDIPM."""

import importlib.util

import cvxpy as cp
import numpy as np
import pytest
import torch
from torch import Tensor

from pinet.torch import QPFunction, project_affine, solve_qp

SEED = 0


def _t(array: object) -> Tensor:
    """Convert an array to a float64 CPU tensor.

    Args:
        array: Array-like value.

    Returns:
        Float64 tensor.
    """
    return torch.tensor(np.asarray(array), dtype=torch.float64)


def _feasible_polytope(
    dim: int, n_eq: int, n_ineq: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a feasible polytope ``A y = b``, ``lb <= C y <= ub``.

    Args:
        dim: Primal dimension.
        n_eq: Number of equalities.
        n_ineq: Number of two-sided inequalities.
        seed: RNG seed.

    Returns:
        Tuple ``(a_mat, b, c_mat, lb, ub)`` without a batch axis.
    """
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(n_eq, dim))
    c_mat = rng.normal(size=(n_ineq, dim))
    x_feas = rng.uniform(-1.0, 1.0, size=(dim,))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.5
    ub = c_mat @ x_feas + 0.5
    return a_mat, b, c_mat, lb, ub


def _two_sided_gh(
    c_mat: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert box inequalities into ``G y <= h``.

    Args:
        c_mat: Two-sided inequality matrix.
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Pair ``(g_mat, h)``.
    """
    g_mat = np.concatenate([c_mat, -c_mat], axis=0)
    h = np.concatenate([ub, -lb], axis=0)
    return g_mat, h


def _cvxpy_qp(
    q_mat: np.ndarray,
    p: np.ndarray,
    g_mat: np.ndarray,
    h: np.ndarray,
    a_mat: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Solve one QP with CVXPY / SCS.

    Args:
        q_mat: Quadratic term.
        p: Linear term.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.
        a_mat: Equality matrix.
        b: Equality right-hand side.

    Returns:
        Primal solution.
    """
    dim = q_mat.shape[0]
    y = cp.Variable(dim)
    constraints = [g_mat @ y <= h]
    if a_mat.shape[0] > 0:
        constraints.append(a_mat @ y == b)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(y, q_mat) + p @ y),
        constraints,
    )
    problem.solve(solver=cp.SCS, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
    assert y.value is not None, f"CVXPY failed with status {problem.status}"
    return np.asarray(y.value)


def _polytope_cv(
    y: np.ndarray,
    a_mat: np.ndarray,
    b: np.ndarray,
    g_mat: np.ndarray,
    h: np.ndarray,
) -> float:
    """Maximum equality / inequality violation.

    Args:
        y: Points ``(B, n)``.
        a_mat: Equality matrix.
        b: Equality right-hand side.
        g_mat: Inequality matrix.
        h: Inequality right-hand side.

    Returns:
        Maximum residual.
    """
    eq = np.max(np.abs(y @ a_mat.T - b), axis=-1) if a_mat.shape[0] > 0 else 0.0
    ineq = np.max(np.maximum(y @ g_mat.T - h, 0.0), axis=-1)
    return float(np.max(np.maximum(eq, ineq)))


def test_project_affine_matches_cvxpy() -> None:
    """Polytope projection matches a high-accuracy CVXPY solve."""
    a_mat, b, c_mat, lb, ub = _feasible_polytope(8, 3, 4, SEED)
    g_mat, h = _two_sided_gh(c_mat, lb, ub)
    rng = np.random.default_rng(SEED + 1)
    x = rng.uniform(-2.0, 2.0, size=(5, 8))
    y = project_affine(_t(x), _t(a_mat), _t(b), _t(g_mat), _t(h))
    y_np = y.detach().numpy()
    for i in range(x.shape[0]):
        y_cp = _cvxpy_qp(np.eye(8), -x[i], g_mat, h, a_mat, b)
        assert np.allclose(y_np[i], y_cp, atol=1e-5, rtol=1e-5), (
            f"Batch item {i}: max abs {np.max(np.abs(y_np[i] - y_cp))}"
        )
    cv = _polytope_cv(y_np, a_mat, b, g_mat, h)
    assert cv < 1e-8, f"Constraint violation {cv} is not qpth-level"


def test_inequality_only_box() -> None:
    """Projection onto a box recovers clipping."""
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 2.0])
    g_mat = np.concatenate([np.eye(2), -np.eye(2)], axis=0)
    h = np.concatenate([ub, -lb], axis=0)
    x = np.array([[3.0, -4.0], [0.5, 0.5], [-2.0, 5.0]])
    y = project_affine(
        _t(x),
        torch.empty(0, 2, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        _t(g_mat),
        _t(h),
    )
    expected = np.clip(x, lb, ub)
    assert torch.allclose(y, _t(expected), atol=1e-8, rtol=1e-8), (
        f"Box projection mismatch: {y.numpy()} vs {expected}"
    )


def test_shared_and_expanded_coefficients_match() -> None:
    """Unbatched ``Q, G, A`` agree with fully expanded coefficients."""
    a_mat, b, c_mat, lb, ub = _feasible_polytope(6, 2, 3, SEED)
    g_mat, h = _two_sided_gh(c_mat, lb, ub)
    rng = np.random.default_rng(SEED + 2)
    x = rng.normal(size=(4, 6))
    q_mat = np.eye(6)
    y_shared = solve_qp(_t(q_mat), _t(-x), _t(g_mat), _t(h), _t(a_mat), _t(b))
    batch = x.shape[0]
    y_expanded = solve_qp(
        _t(np.broadcast_to(q_mat, (batch, 6, 6)).copy()),
        _t(-x),
        _t(np.broadcast_to(g_mat, (batch, *g_mat.shape)).copy()),
        _t(np.broadcast_to(h, (batch, *h.shape)).copy()),
        _t(np.broadcast_to(a_mat, (batch, *a_mat.shape)).copy()),
        _t(np.broadcast_to(b, (batch, *b.shape)).copy()),
    )
    assert torch.allclose(y_shared, y_expanded, atol=1e-10, rtol=1e-10), (
        "Shared and expanded coefficient layouts must agree."
    )


def test_general_spd_quadratic() -> None:
    """A non-identity SPD ``Q`` matches CVXPY."""
    rng = np.random.default_rng(SEED + 3)
    dim = 5
    factor = rng.normal(size=(dim, dim))
    q_mat = factor.T @ factor + 0.1 * np.eye(dim)
    g_mat = rng.normal(size=(4, dim))
    x_feas = rng.normal(size=(dim,))
    h = g_mat @ x_feas + 0.8
    a_mat = rng.normal(size=(2, dim))
    b = a_mat @ x_feas
    p = rng.normal(size=(dim,))
    y = solve_qp(_t(q_mat), _t(p), _t(g_mat), _t(h), _t(a_mat), _t(b))
    y_cp = _cvxpy_qp(q_mat, p, g_mat, h, a_mat, b)
    assert np.allclose(y.detach().numpy()[0], y_cp, atol=1e-5, rtol=1e-5), (
        f"SPD QP mismatch: max abs {np.max(np.abs(y.detach().numpy()[0] - y_cp))}"
    )


def test_gradient_matches_finite_difference() -> None:
    """d(loss)/dp from the KKT backward matches central differences."""
    a_mat, b, c_mat, lb, ub = _feasible_polytope(6, 2, 3, SEED)
    g_mat, h = _two_sided_gh(c_mat, lb, ub)
    rng = np.random.default_rng(SEED + 4)
    p0 = _t(-rng.normal(size=(3, 6))).requires_grad_(True)
    vec = _t(rng.normal(size=(3, 6)))

    def loss_at(p: Tensor) -> Tensor:
        y = QPFunction(max_iter=20)(_t(np.eye(6)), p, _t(g_mat), _t(h), _t(a_mat), _t(b))
        return (y * vec).sum()

    loss = loss_at(p0)
    (grad,) = torch.autograd.grad(loss, p0)
    direction = _t(rng.normal(size=p0.shape))
    direction = direction / torch.linalg.vector_norm(direction)
    eps = 1e-5
    plus = loss_at(p0.detach() + eps * direction)
    minus = loss_at(p0.detach() - eps * direction)
    fd = (plus - minus) / (2 * eps)
    directional = (grad * direction).sum()
    assert torch.allclose(directional, fd, atol=1e-4, rtol=1e-3), (
        f"FD {fd.item()} vs analytic {directional.item()}"
    )


def test_qpfunction_rejects_empty_inequalities() -> None:
    """Inequality-free QPs are rejected with a clear error."""
    q_mat = torch.eye(3, dtype=torch.float64)
    p = torch.zeros(3, dtype=torch.float64)
    g_mat = torch.empty(0, 3, dtype=torch.float64)
    h = torch.empty(0, dtype=torch.float64)
    a_mat = torch.eye(3, dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="inequality"):
        solve_qp(q_mat, p, g_mat, h, a_mat, b)


@pytest.mark.skipif(
    importlib.util.find_spec("qpth") is None,
    reason="qpth is optional (pip install qpth --no-deps)",
)
def test_matches_qpth_when_installed() -> None:
    """Solutions agree with locuslab/qpth when that package is present."""
    qpth_qp = pytest.importorskip("qpth.qp")
    a_mat, b, c_mat, lb, ub = _feasible_polytope(6, 2, 3, SEED)
    g_mat, h = _two_sided_gh(c_mat, lb, ub)
    rng = np.random.default_rng(SEED + 5)
    x = rng.uniform(-2.0, 2.0, size=(4, 6))
    q_mat = torch.eye(6, dtype=torch.float64)
    y_fast = solve_qp(q_mat, _t(-x), _t(g_mat), _t(h), _t(a_mat), _t(b))
    y_qpth = qpth_qp.QPFunction(verbose=False)(
        q_mat, _t(-x), _t(g_mat), _t(h), _t(a_mat), _t(b)
    )
    assert torch.allclose(y_fast, y_qpth, atol=1e-5, rtol=1e-5), (
        f"qpth mismatch: max abs {(y_fast - y_qpth).abs().max().item()}"
    )
