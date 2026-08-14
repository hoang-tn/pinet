"""API tests for the Torch-native projection layer."""

import numpy as np
import torch

from pinet.torch import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    Project,
)


def test_vector_layout() -> None:
    """A 1D input returns a 1D projection."""
    proj = Project(
        box_constraint=BoxConstraint(lb=np.array([-1.0, -1.0]), ub=np.array([1.0, 1.0])),
        compile=False,
    )
    x = torch.tensor([3.0, -4.0], dtype=torch.float64)
    y = proj(x)
    assert y.shape == (2,), f"Expected (2,), got {tuple(y.shape)}"
    assert torch.allclose(y, torch.tensor([1.0, -1.0], dtype=torch.float64))


def test_return_state_and_warm_start() -> None:
    """Warm-starting from ``sK`` continues the ADMM sequence."""
    rng = np.random.default_rng(0)
    dim = 6
    n_eq = 2
    n_ineq = 3
    a_mat = rng.normal(size=(1, n_eq, dim))
    c_mat = rng.normal(size=(1, n_ineq, dim))
    x_feas = rng.uniform(-1, 1, size=(1, dim, 1))
    b = a_mat @ x_feas
    lb = c_mat @ x_feas - 0.5
    ub = c_mat @ x_feas + 0.5
    x = torch.tensor(rng.normal(size=(2, dim)), dtype=torch.float64)
    proj = Project(
        EqualityConstraint(a_mat, b),
        AffineInequalityConstraint(c_mat, lb, ub),
        n_iter=10,
        compile=False,
    )
    y_short, s_k = proj(x, n_iter=10, return_state=True)
    y_warm = proj(x, s0=s_k, n_iter=10)
    y_long = proj(x, n_iter=20)
    assert torch.allclose(y_warm, y_long, atol=1e-8, rtol=1e-8), (
        "Warm-start of 10+10 iterations should match a single 20-iteration run."
    )
    assert y_short.shape == x.shape, "Projected point should match the input layout."


def test_cv_shape() -> None:
    """Constraint violation is a batch vector for 2D inputs."""
    proj = Project(
        box_constraint=BoxConstraint(lb=np.array([-1.0, -1.0]), ub=np.array([1.0, 1.0])),
        compile=False,
    )
    x = torch.tensor([[2.0, 0.0], [0.0, -3.0]], dtype=torch.float64)
    cv = proj.cv(x)
    assert cv.shape == (2,), f"Expected (2,), got {tuple(cv.shape)}"
    assert torch.allclose(cv, torch.tensor([1.0, 2.0], dtype=torch.float64))
