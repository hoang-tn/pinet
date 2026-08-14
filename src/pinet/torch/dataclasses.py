"""Dataclasses used by the Torch projection layer."""

from dataclasses import dataclass, replace

from torch import Tensor

from pinet.constants import Constants
from pinet.constraints.non_linear_types import (
    L2NormType,
    NonLinearConstraintType,
    SOCType,
)


@dataclass
class EquilibrationParams:
    """Parameters for modified Ruiz equilibration.

    Attributes:
        max_iter: Maximum number of equilibration iterations.
        tol: Tolerance for convergence of the equilibration.
        ord: Order of the norm used for the convergence check.
        col_scaling: Whether to apply column scaling.
        update_mode: ``"Gauss"`` or ``"Jacobi"``.
        safeguard: Check that the condition number of ``a_mat`` does not increase.
    """

    max_iter: int = Constants.EQUILIBRATION_DEFAULT_MAX_ITER
    tol: float = Constants.EQUILIBRATION_DEFAULT_TOL
    ord: float = Constants.EQUILIBRATION_DEFAULT_ORD
    col_scaling: bool = Constants.EQUILIBRATION_DEFAULT_COL_SCALING
    update_mode: str = Constants.EQUILIBRATION_DEFAULT_UPDATE_MODE
    safeguard: bool = Constants.EQUILIBRATION_DEFAULT_SAFEGUARD

    def validate(self) -> None:
        """Validate the equilibration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.ord not in [1, 2, float("inf")]:
            raise ValueError("ord must be 1, 2, or infinity.")
        if self.update_mode not in ["Gauss", "Jacobi"]:
            raise ValueError('update_mode must be either "Gauss" or "Jacobi".')

    def update(self, **kwargs: object) -> "EquilibrationParams":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)


@dataclass
class NonLinearSpecification:
    """Specification of a non-linear constraint ``g(A y + a) <= f y + b``.

    Attributes:
        nl_type: Non-linear constraint type (SOC or L2).
        a_mat: Linear map before the cone, shape ``(1, m, n)``.
        a: Optional offset for the cone vector.
        f: Optional linear map for the cone scalar.
        b: Optional scalar offset for the cone bound.
    """

    nl_type: NonLinearConstraintType
    a_mat: Tensor
    a: Tensor | None = None
    f: Tensor | None = None
    b: Tensor | None = None

    def update(self, **kwargs: object) -> "NonLinearSpecification":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)

    def validate(self) -> None:
        """Validate the supported non-linear type.

        Raises:
            ValueError: If ``nl_type`` is not SOC or L2.
        """
        if self.nl_type == L2NormType and self.f is not None:
            raise ValueError(
                "L2NormType with RHS (f) is not supported in NonLinearSpecification. "
                "Use SOCType instead."
            )
        if self.nl_type not in (SOCType, L2NormType):
            raise ValueError(
                f"nl_type must be SOCType or L2NormType, got {self.nl_type!r}."
            )
