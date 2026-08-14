"""Affine inequality constraint module (Torch)."""

from typing import Any

from torch import Tensor

from pinet.torch._ops import ineq_cv
from pinet.torch._shapes import TensorLike, as_col, as_matrix

from .base import Constraint, set_buffer


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint ``lb <= c_mat @ x <= ub``.

    Attributes:
        c_mat: Inequality matrix, shape ``(B, n_ineq, n)``.
        lb: Lower bound, shape ``(B, n_ineq, 1)``.
        ub: Upper bound, shape ``(B, n_ineq, 1)``.
    """

    c_mat: Tensor
    lb: Tensor
    ub: Tensor

    def __init__(self, c_mat: TensorLike, lb: TensorLike, ub: TensorLike) -> None:
        """Initialize the affine inequality constraint.

        Args:
            c_mat: Inequality matrix, shape ``(n_ineq, n)`` or ``(B, n_ineq, n)``.
            lb: Lower bound, shape ``(n_ineq,)``, ``(B, n_ineq)``, or ``(B, n_ineq, 1)``.
            ub: Upper bound, matching ``lb``.
        """
        super().__init__()
        c_mat_t = as_matrix(c_mat)
        lb_t = as_col(lb, dtype=c_mat_t.dtype, device=c_mat_t.device)
        ub_t = as_col(ub, dtype=c_mat_t.dtype, device=c_mat_t.device)
        assert (
            c_mat_t.shape[0] == lb_t.shape[0]
            or c_mat_t.shape[0] == 1
            or lb_t.shape[0] == 1
        ), (
            "Batch sizes are inconsistent: "
            f"c_mat{tuple(c_mat_t.shape)}, l{tuple(lb_t.shape)}"
        )
        assert (
            c_mat_t.shape[0] == ub_t.shape[0]
            or c_mat_t.shape[0] == 1
            or ub_t.shape[0] == 1
        ), (
            f"Batch sizes are inconsistent: c_mat{tuple(c_mat_t.shape)}, "
            f"ub{tuple(ub_t.shape)}"
        )
        assert c_mat_t.shape[1] == lb_t.shape[1], (
            "Number of rows in c_mat must equal size of l."
        )
        assert c_mat_t.shape[1] == ub_t.shape[1], (
            "Number of rows in c_mat must equal size of u."
        )
        self.c_mat = set_buffer(self, "c_mat", c_mat_t)
        self.lb = set_buffer(self, "lb", lb_t)
        self.ub = set_buffer(self, "ub", ub_t)

    def project(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Project onto the affine inequality set.

        Args:
            x: Point to project.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.

        Raises:
            NotImplementedError: Always. Inequalities are lifted, not projected.
        """
        del x, kwargs
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    def cv(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Compute the inequality violation.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.
        """
        del kwargs
        return ineq_cv(x, self.c_mat, self.lb, self.ub)

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return int(self.c_mat.shape[-1])

    @property
    def n_constraints(self) -> int:
        """Return the number of inequality rows."""
        return int(self.c_mat.shape[1])
