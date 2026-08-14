"""Compiled vs eager agreement for the Torch projector."""

import numpy as np
import pytest
import torch

from pinet.torch import AffineInequalityConstraint, EqualityConstraint, Project


def _small_polytope() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build a tiny feasible eq+ineq instance.

    Returns:
        Tuple ``(a_mat, b, c_mat, lb, ub)``.
    """
    rng = np.random.default_rng(0)
    dim = 6
    n_eq = 2
    n_ineq = 2
    a_mat = rng.normal(size=(1, n_eq, dim)).astype(np.float32)
    c_mat = rng.normal(size=(1, n_ineq, dim)).astype(np.float32)
    x_feas = rng.uniform(-1, 1, size=(1, dim, 1)).astype(np.float32)
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.5
    ub = c_mat @ x_feas + 0.5
    return a_mat, b, c_mat, lb, ub


def test_compiled_admm_matches_eager() -> None:
    """Dynamo-captured ADMM matches the eager loop on a small polytope."""
    a_mat, b, c_mat, lb, ub = _small_polytope()
    x = torch.randn(2, 6, dtype=torch.float32)
    eq = EqualityConstraint(a_mat, b)
    ineq = AffineInequalityConstraint(c_mat, lb, ub)
    eager = Project(eq, ineq, n_iter=8, compile=False)
    compiled = Project(
        eq, ineq, n_iter=8, compile=True, compile_mode="default", compile_backend="eager"
    )
    y_eager = eager(x)
    y_compiled = compiled(x)
    assert torch.allclose(y_eager, y_compiled, atol=1e-4, rtol=1e-4), (
        "Compiled ADMM should match eager."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_admm_matches_eager_cuda() -> None:
    """Inductor-compiled ADMM matches eager on CUDA."""
    a_mat, b, c_mat, lb, ub = _small_polytope()
    x = torch.randn(4, 6, dtype=torch.float32, device="cuda")
    eq = EqualityConstraint(a_mat, b)
    ineq = AffineInequalityConstraint(c_mat, lb, ub)
    eager = Project(eq, ineq, n_iter=20, compile=False).cuda()
    compiled = Project(eq, ineq, n_iter=20, compile=True, compile_mode="default").cuda()
    y_eager = eager(x)
    y_compiled = compiled(x)
    assert torch.allclose(y_eager, y_compiled, atol=1e-4, rtol=1e-4), (
        "Compiled ADMM should match eager on CUDA."
    )
