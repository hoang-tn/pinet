"""Equality constraint module (Torch)."""

from typing import Any

import torch
from torch import Tensor

from pinet.torch._ops import equality_cv, equality_project
from pinet.torch._shapes import TensorLike, as_col, as_matrix

from .base import Constraint, set_buffer

_EXPECTED_NDIM = 3
_LAST_AXIS_SIZE_B = 1


class EqualityConstraint(Constraint):
    """Affine equality constraint ``a_mat @ x == b``.

    Attributes:
        a_mat: Left-hand side matrix, shape ``(B, m, n)``.
        b: Right-hand side vector, shape ``(B, m, 1)``.
        a_mat_pinv: Cached pseudoinverse when ``method="pinv"`` and ``var_a_mat``
            is false.
        method: Solver strategy; ``"pinv"`` or ``None``.
        var_b: Whether ``b`` is expected to vary per instance.
        var_a_mat: Whether ``a_mat`` is expected to vary per instance.
    """

    a_mat: Tensor
    b: Tensor
    a_mat_pinv: Tensor | None
    method: str | None
    var_b: bool
    var_a_mat: bool

    def __init__(
        self,
        a_mat: TensorLike,
        b: TensorLike,
        method: str | None = "pinv",
        var_b: bool | None = False,
        var_a_mat: bool | None = False,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            a_mat: Left-hand side matrix, shape ``(m, n)`` or ``(B, m, n)``.
            b: Right-hand side vector, shape ``(m,)``, ``(B, m)``, or ``(B, m, 1)``.
            method: Linear-algebra method; ``"pinv"`` or ``None``.
            var_b: Whether ``b`` changes per instance.
            var_a_mat: Whether ``a_mat`` changes per instance.
        """
        super().__init__()
        a_mat_t = as_matrix(a_mat)
        b_t = as_col(b, dtype=a_mat_t.dtype, device=a_mat_t.device)
        assert a_mat_t.ndim == _EXPECTED_NDIM, (
            "a_mat is a matrix with shape (batch_size, n_constraints, dimension)."
        )
        assert b_t.ndim == _EXPECTED_NDIM, (
            "b is a matrix with shape (batch_size, n_constraints, 1)."
        )
        assert b_t.shape[2] == _LAST_AXIS_SIZE_B, (
            "b must have shape (batch_size, n_constraints, 1)."
        )
        assert (
            a_mat_t.shape[0] == b_t.shape[0] or a_mat_t.shape[0] == 1 or b_t.shape[0] == 1
        ), (
            "Batch sizes are inconsistent: "
            f"a_mat{tuple(a_mat_t.shape)}, b{tuple(b_t.shape)}"
        )
        assert a_mat_t.shape[1] == b_t.shape[1], (
            "Number of rows in a_mat must equal size of b."
        )

        valid_methods = ["pinv", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method {method}. Valid methods are: {valid_methods}"
            )

        self.method = method
        self.var_b = bool(var_b)
        self.var_a_mat = bool(var_a_mat)
        self.a_mat = set_buffer(self, "a_mat", a_mat_t)
        self.b = set_buffer(self, "b", b_t)
        if method == "pinv":
            self.a_mat_pinv = set_buffer(self, "a_mat_pinv", torch_pinv(a_mat_t))
        else:
            self.a_mat_pinv = None

    def resolve(
        self,
        x: Tensor,
        b: Tensor | None = None,
        a_mat: Tensor | None = None,
        a_mat_pinv: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Resolve runtime equality parameters.

        Args:
            x: Point being projected, used for dtype/device.
            b: Optional runtime right-hand side.
            a_mat: Optional runtime matrix.
            a_mat_pinv: Optional runtime pseudoinverse.

        Returns:
            Tuple ``(b, a_mat, a_mat_pinv)``.
        """
        b_use = self.b if b is None else b
        a_use = self.a_mat if (a_mat is None or not self.var_a_mat) else a_mat
        pinv_use = self.a_mat_pinv if a_mat_pinv is None else a_mat_pinv
        if pinv_use is None:
            pinv_use = torch_pinv(a_use)
        return (
            b_use.to(dtype=x.dtype, device=x.device),
            a_use.to(dtype=x.dtype, device=x.device),
            pinv_use.to(dtype=x.dtype, device=x.device),
        )

    def project(
        self,
        x: Tensor,
        b: Tensor | None = None,
        a_mat: Tensor | None = None,
        a_mat_pinv: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Project onto the equality set.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            b: Optional runtime right-hand side.
            a_mat: Optional runtime matrix.
            a_mat_pinv: Optional runtime pseudoinverse.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.

        Raises:
            NotImplementedError: If ``method`` is ``None``.
        """
        if self.method is None:
            raise NotImplementedError("No projection method set.")
        del kwargs
        b_use, a_use, pinv_use = self.resolve(x, b=b, a_mat=a_mat, a_mat_pinv=a_mat_pinv)
        return equality_project(x, a_use, pinv_use, b_use)

    def cv(
        self,
        x: Tensor,
        b: Tensor | None = None,
        a_mat: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the equality residual.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            b: Optional runtime right-hand side.
            a_mat: Optional runtime matrix.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.
        """
        del kwargs
        b_use, a_use, _ = self.resolve(x, b=b, a_mat=a_mat)
        return equality_cv(x, a_use, b_use)

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return int(self.a_mat.shape[-1])

    @property
    def n_constraints(self) -> int:
        """Return the number of equality rows."""
        return int(self.a_mat.shape[1])


def torch_pinv(a_mat: Tensor) -> Tensor:
    """Batched Moore-Penrose pseudoinverse.

    Args:
        a_mat: Matrix of shape ``(B, m, n)``.

    Returns:
        Pseudoinverse of shape ``(B, n, m)``.
    """
    if a_mat.shape[1] == 0:
        batch, _, n = a_mat.shape
        return a_mat.new_zeros(batch, n, 0)
    return torch.linalg.pinv(a_mat)
