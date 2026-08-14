"""Second-order cone constraint module (Torch)."""

from typing import Any

import torch
from torch import Tensor

from pinet.constants import Constants
from pinet.torch._ops import soc_cv, soc_project
from pinet.torch._shapes import TensorLike, as_col, as_tensor, scatter_to_full

from .base import Constraint, set_buffer

SOC_CONSTRAINT_EPSILON = Constants.SOC_CONSTRAINT_EPSILON


class SocConstraint(Constraint):
    """Second-order cone ``||x[mask_u] + a||_2 <= x[mask_t] + b``.

    Attributes:
        mask_u: Boolean mask for the cone vector.
        mask_t: Boolean mask for the cone scalar.
        a: Offset added to the cone vector.
        b: Scalar offset added to the cone bound.
        eps: Stabilizer for the projection direction.
    """

    mask_u: Tensor
    mask_t: Tensor
    a: Tensor
    b: Tensor
    eps: float

    def __init__(
        self,
        mask_u: TensorLike,
        mask_t: TensorLike,
        a: TensorLike | None = None,
        b: TensorLike | None = None,
        eps: float = SOC_CONSTRAINT_EPSILON,
    ) -> None:
        """Initialize the SOC constraint.

        Args:
            mask_u: Boolean mask selecting the cone vector coordinates.
            mask_t: Boolean mask selecting the cone scalar; must have one True.
            a: Optional offset for the cone vector.
            b: Optional scalar offset for the cone bound.
            eps: Stabilizer for ``u / ||u||``.
        """
        super().__init__()
        mask_u_t = as_tensor(mask_u).reshape(-1).to(dtype=torch.bool)
        mask_t_t = as_tensor(mask_t).reshape(-1).to(dtype=torch.bool)
        if mask_u_t.numel() != mask_t_t.numel():
            raise ValueError("mask_u and mask_t must have the same length.")
        if int(mask_t_t.sum()) != 1:
            raise ValueError("mask_t must select exactly one element.")
        dim = int(mask_u_t.numel())
        n_u = int(mask_u_t.sum())
        if a is None:
            a_t = torch.zeros((1, n_u, 1), dtype=torch.float32)
        else:
            a_t = as_col(a)
        if b is None:
            b_t = torch.zeros((1, 1, 1), dtype=a_t.dtype, device=a_t.device)
        else:
            b_t = as_col(b, dtype=a_t.dtype, device=a_t.device)
        if int(a_t.shape[1]) != n_u:
            raise ValueError(
                "The second dimension of a must match the number of True values "
                f"in mask_u. Received a shape: {tuple(a_t.shape)}, "
                f"mask_u has {n_u} True values."
            )
        a_full = scatter_to_full(a_t, mask_u_t, dim)
        b_full = scatter_to_full(b_t, mask_t_t, dim)
        self._dim = dim
        self.eps = eps
        self.mask_u = set_buffer(self, "mask_u", mask_u_t.view(1, dim, 1))
        self.mask_t = set_buffer(self, "mask_t", mask_t_t.view(1, dim, 1))
        self.a = set_buffer(self, "a", a_t)
        self.b = set_buffer(self, "b", b_t)
        self.a_full = set_buffer(self, "a_full", a_full)
        self.b_full = set_buffer(self, "b_full", b_full)
        self.eps_t = set_buffer(self, "eps_t", torch.tensor(eps, dtype=a_t.dtype))

    def resolve_offsets(
        self, x: Tensor, a: Tensor | None = None, b: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Resolve dense offsets and the stabilizer tensor.

        Args:
            x: Point being projected, used for dtype/device.
            a: Optional runtime cone-vector offset.
            b: Optional runtime scalar offset.

        Returns:
            Tuple ``(a_full, b_full, eps)``.
        """
        dim = self._dim
        if a is None:
            a_full = self.a_full.to(dtype=x.dtype, device=x.device)
        else:
            a_t = as_col(a, dtype=x.dtype, device=x.device)
            a_full = scatter_to_full(a_t, self.mask_u.reshape(-1), dim)
        if b is None:
            b_full = self.b_full.to(dtype=x.dtype, device=x.device)
        else:
            b_t = as_col(b, dtype=x.dtype, device=x.device)
            b_full = scatter_to_full(b_t, self.mask_t.reshape(-1), dim)
        eps = self.eps_t.to(dtype=x.dtype, device=x.device)
        return a_full, b_full, eps

    def project(
        self, x: Tensor, a: Tensor | None = None, b: Tensor | None = None, **kwargs: Any
    ) -> Tensor:
        """Project onto the second-order cone.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            a: Optional runtime cone-vector offset.
            b: Optional runtime scalar offset.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.
        """
        del kwargs
        a_full, b_full, eps = self.resolve_offsets(x, a=a, b=b)
        return soc_project(x, self.mask_u, self.mask_t, a_full, b_full, eps)

    def cv(
        self, x: Tensor, a: Tensor | None = None, b: Tensor | None = None, **kwargs: Any
    ) -> Tensor:
        """Compute the SOC violation.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            a: Optional runtime cone-vector offset.
            b: Optional runtime scalar offset.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.
        """
        del kwargs
        a_full, b_full, _ = self.resolve_offsets(x, a=a, b=b)
        return soc_cv(x, self.mask_u, self.mask_t, a_full, b_full)

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of SOC constraints (always 1)."""
        return 1
