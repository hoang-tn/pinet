"""Torch backend for the Πnet orthogonal projection layer.

Import this submodule for a ``torch.nn.Module`` API:

.. code-block:: python

    from pinet.torch import Project, EqualityConstraint, BoxConstraint
    y = Project(eq, box)(x)

A batched PDIPM (``QPFunction`` / ``solve_qp``) is also available as a
faster qpth-compatible quadratic-program layer.
"""

from pinet.constraints.non_linear_types import (
    L2NormType,
    NonLinearConstraintType,
    SOCType,
)
from pinet.torch.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    CartesianConstraint,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
    SocConstraint,
)
from pinet.torch.dataclasses import EquilibrationParams, NonLinearSpecification
from pinet.torch.equilibration import ruiz_equilibration
from pinet.torch.project import Project
from pinet.torch.qp import QPFunction, project_affine, solve_qp
from pinet.torch.solver import iteration_step

__all__ = [
    "AffineInequalityConstraint",
    "BoxConstraint",
    "CartesianConstraint",
    "ConstraintParser",
    "EqualityConstraint",
    "EquilibrationParams",
    "L2NormType",
    "NonLinearConstraint",
    "NonLinearConstraintType",
    "NonLinearSpecification",
    "Project",
    "QPFunction",
    "SOCType",
    "SocConstraint",
    "iteration_step",
    "project_affine",
    "ruiz_equilibration",
    "solve_qp",
]
