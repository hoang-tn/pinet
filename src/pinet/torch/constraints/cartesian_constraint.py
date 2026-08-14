"""Cartesian product of box and SOC constraints (Torch)."""

from typing import Any

import torch
from torch import Tensor

from .base import Constraint
from .box import BoxConstraint
from .soc_constraint import SocConstraint


class CartesianConstraint(Constraint):
    """Cartesian product of a box and one or more SOC constraints.

    Masks of the constituent constraints must be disjoint.
    """

    def __init__(
        self,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[SocConstraint] | None = None,
    ) -> None:
        """Initialize the Cartesian constraint.

        Args:
            box_constraint: Optional box component.
            nl_constraints: Optional list of SOC components.

        Raises:
            ValueError: If no constraints are provided, dimensions disagree,
                or masks overlap.
        """
        super().__init__()
        nls = [] if nl_constraints is None else list(nl_constraints)
        self.box_constraint = box_constraint
        self.nl_modules = torch.nn.ModuleList(nls)
        self.n_nonlinear = len(nls)
        self._dim = _validate_constraints(box_constraint, nls)

    def _socs(self) -> list[SocConstraint]:
        """Return SOC constituents with a precise type.

        Returns:
            SOC constraints stored on this product.
        """
        return [soc for soc in self.nl_modules if isinstance(soc, SocConstraint)]

    @property
    def nl_constraints(self) -> list[SocConstraint]:
        """Return the SOC constituents.

        Returns:
            SOC constraints stored on this product.
        """
        return self._socs()

    @property
    def constraints(self) -> list[Constraint]:
        """Return constituent constraints in order."""
        out: list[Constraint] = []
        if self.box_constraint is not None:
            out.append(self.box_constraint)
        out.extend(self._socs())
        return out

    def project(
        self,
        x: Tensor,
        box_lb: Tensor | None = None,
        box_ub: Tensor | None = None,
        nl_a: list[Tensor] | None = None,
        nl_b: list[Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Project onto each constituent constraint in turn.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-SOC vector offsets.
            nl_b: Optional per-SOC scalar offsets.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.
        """
        del kwargs
        y = x
        if self.box_constraint is not None:
            y = self.box_constraint.project(y, lb=box_lb, ub=box_ub)
        for i, soc in enumerate(self._socs()):
            a_i = None if nl_a is None else nl_a[i]
            b_i = None if nl_b is None else nl_b[i]
            y = soc.project(y, a=a_i, b=b_i)
        return y

    def cv(
        self,
        x: Tensor,
        box_lb: Tensor | None = None,
        box_ub: Tensor | None = None,
        nl_a: list[Tensor] | None = None,
        nl_b: list[Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Maximum violation across constituent constraints.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-SOC vector offsets.
            nl_b: Optional per-SOC scalar offsets.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.
        """
        del kwargs
        cvs: list[Tensor] = []
        if self.box_constraint is not None:
            cvs.append(self.box_constraint.cv(x, lb=box_lb, ub=box_ub))
        for i, soc in enumerate(self._socs()):
            a_i = None if nl_a is None else nl_a[i]
            b_i = None if nl_b is None else nl_b[i]
            cvs.append(soc.cv(x, a=a_i, b=b_i))
        stacked = torch.stack(cvs, dim=0)
        return stacked.amax(dim=0)

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the total number of constituent constraints."""
        return sum(c.n_constraints for c in self.constraints)


def _validate_constraints(
    box_constraint: BoxConstraint | None,
    nl_constraints: list[SocConstraint],
) -> int:
    """Validate types, dimensions, and disjoint masks.

    Args:
        box_constraint: Optional box component.
        nl_constraints: SOC components.

    Returns:
        Shared primal dimension.

    Raises:
        ValueError: If validation fails.
    """
    constraints: list[Constraint] = [
        c for c in (box_constraint, *nl_constraints) if c is not None
    ]
    if not constraints:
        raise ValueError("At least one constraint must be provided.")
    dim = constraints[0].dim
    for constraint in constraints:
        if constraint.dim != dim:
            raise ValueError(
                f"All constraints must have the same dimension. "
                f"Expected {dim}, got {constraint.dim}."
            )
    used = torch.zeros(dim, dtype=torch.bool)
    for constraint in constraints:
        if isinstance(constraint, BoxConstraint):
            new_mask = constraint.mask.reshape(-1).to(dtype=torch.bool)
        else:
            assert isinstance(constraint, SocConstraint)
            new_mask = constraint.mask_u.reshape(-1) | constraint.mask_t.reshape(-1)
        if bool(torch.any(used & new_mask)):
            raise ValueError(
                "Constraint masks overlap with previously defined constraints."
            )
        used = used | new_mask
    return dim
