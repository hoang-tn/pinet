"""Generic non-linear constraint carrier (Torch)."""

from typing import Any

from torch import Tensor

from pinet.constraints.non_linear_types import NonLinearConstraintType
from pinet.torch._shapes import TensorLike, as_col, as_matrix
from pinet.torch.dataclasses import NonLinearSpecification

from .base import Constraint, set_buffer


class NonLinearConstraint(Constraint):
    """Parameter carrier for ``g(A y + a) <= f y + b``.

    Projection is performed after lifting onto a primitive (currently SOC).
    """

    def __init__(
        self,
        spec: NonLinearSpecification | None = None,
        *,
        a_mat: TensorLike | None = None,
        a: TensorLike | None = None,
        f: TensorLike | None = None,
        b: TensorLike | None = None,
        nl_type: NonLinearConstraintType | None = None,
    ) -> None:
        """Initialize the non-linear constraint.

        Args:
            spec: Optional full specification object.
            a_mat: Linear map before the cone, used when ``spec`` is omitted.
            a: Optional cone-vector offset.
            f: Optional linear map for the cone scalar.
            b: Optional scalar offset.
            nl_type: Non-linear type, used when ``spec`` is omitted.
        """
        super().__init__()
        if spec is None:
            if a_mat is None or nl_type is None:
                raise ValueError("Provide spec, or both a_mat and nl_type.")
            spec = NonLinearSpecification(
                nl_type=nl_type,
                a_mat=as_matrix(a_mat),
                a=None if a is None else as_col(a),
                f=None if f is None else as_matrix(f),
                b=None if b is None else as_col(b),
            )
        spec.validate()
        self.nl_type = spec.nl_type
        self._dim = int(spec.a_mat.shape[-1])
        self.a_mat = set_buffer(self, "a_mat", spec.a_mat)
        if spec.a is None:
            self.a = None
        else:
            self.a = set_buffer(self, "a", spec.a)
        if spec.f is None:
            self.f = None
        else:
            self.f = set_buffer(self, "f", spec.f)
        if spec.b is None:
            self.b = None
        else:
            self.b = set_buffer(self, "b", spec.b)
        self.spec = spec

    def project(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Project onto the non-linear set.

        Args:
            x: Point to project.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.

        Raises:
            NotImplementedError: Always; use the lifted primitive instead.
        """
        del x, kwargs
        raise NotImplementedError(
            "NonLinearConstraint is a parameter carrier; use a concrete subclass."
        )

    def cv(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Compute the constraint violation.

        Args:
            x: Point to evaluate.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation.

        Raises:
            NotImplementedError: Always; use the lifted primitive instead.
        """
        del x, kwargs
        raise NotImplementedError(
            "NonLinearConstraint is a parameter carrier; use a concrete subclass."
        )

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of non-linear constraints (always 1)."""
        return 1
