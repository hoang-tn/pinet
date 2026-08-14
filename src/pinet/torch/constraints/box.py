"""Box constraint module (Torch)."""

import math
from typing import Any

import torch
from torch import Tensor

from pinet.torch._ops import box_cv, box_project
from pinet.torch._shapes import TensorLike, as_col, as_tensor, scatter_to_full

from .base import Constraint, set_buffer


class BoxConstraint(Constraint):
    """Box constraint on a subset of coordinates.

    Attributes:
        lb: Lower bound on the active coordinates, or ``None``.
        ub: Upper bound on the active coordinates, or ``None``.
        mask: Boolean mask of length ``dim``.
        lb_full: Dense lower bound with ``-inf`` on free coordinates.
        ub_full: Dense upper bound with ``inf`` on free coordinates.
        scale: Scaling applied to runtime bounds.
    """

    lb: Tensor
    ub: Tensor
    mask: Tensor
    lb_full: Tensor
    ub_full: Tensor
    scale: Tensor

    def __init__(
        self,
        lb: TensorLike | None = None,
        ub: TensorLike | None = None,
        mask: TensorLike | None = None,
        scale: TensorLike | None = None,
    ) -> None:
        """Initialize the box constraint.

        Args:
            lb: Lower bound, shape ``(n,)``, ``(B, n)``, or ``(B, n, 1)``.
            ub: Upper bound, matching ``lb``.
            mask: Optional boolean mask selecting constrained coordinates.
            scale: Optional scaling applied to runtime bounds.
        """
        super().__init__()
        if lb is None and ub is None:
            raise ValueError("At least one of lower or upper bounds must be provided.")

        mask_t: Tensor | None
        if mask is not None:
            mask_t = as_tensor(mask).reshape(-1).to(dtype=torch.bool)
            dim = int(mask_t.numel())
        else:
            mask_t = None
            sample = lb if lb is not None else ub
            assert sample is not None
            dim = int(as_col(sample).shape[1])

        lb_t = as_col(lb) if lb is not None else None
        ub_t = as_col(ub) if ub is not None else None
        if lb_t is not None and ub_t is not None:
            ub_t = ub_t.to(dtype=lb_t.dtype, device=lb_t.device)
            if not bool(torch.all(lb_t <= ub_t)):
                raise ValueError(
                    "Lower bound must be less than or equal to the upper bound."
                )
        if lb_t is not None:
            dtype = lb_t.dtype
            device = lb_t.device
            n_active = int(lb_t.shape[1])
        else:
            assert ub_t is not None
            dtype = ub_t.dtype
            device = ub_t.device
            n_active = int(ub_t.shape[1])

        if mask_t is None:
            mask_t = torch.ones(dim, dtype=torch.bool, device=device)
        else:
            mask_t = mask_t.to(device=device)
            if int(mask_t.sum()) != n_active:
                raise ValueError(
                    "Number of active entries in the mask must match the bounds. "
                    f"Received mask shape: {tuple(mask_t.shape)}, "
                    f"n_active: {n_active}."
                )

        if lb_t is None:
            assert ub_t is not None
            lb_t = torch.full_like(ub_t, -math.inf)
        if ub_t is None:
            ub_t = torch.full_like(lb_t, math.inf)

        if scale is None:
            scale_t = torch.ones((1, n_active, 1), dtype=dtype, device=device)
        else:
            scale_t = as_col(scale, dtype=dtype, device=device)

        lb_full = _dense_bounds(lb_t, mask_t, dim, fill=-math.inf)
        ub_full = _dense_bounds(ub_t, mask_t, dim, fill=math.inf)

        self._dim = dim
        self._n_constraints = n_active
        self.lb = set_buffer(self, "lb", lb_t)
        self.ub = set_buffer(self, "ub", ub_t)
        self.mask = set_buffer(self, "mask", mask_t)
        self.scale = set_buffer(self, "scale", scale_t)
        self.lb_full = set_buffer(self, "lb_full", lb_full)
        self.ub_full = set_buffer(self, "ub_full", ub_full)

    def resolve_bounds(
        self, x: Tensor, lb: Tensor | None = None, ub: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Resolve dense bounds, applying scale to runtime values.

        Args:
            x: Point being projected, used for dtype/device.
            lb: Optional runtime lower bound on the active coordinates.
            ub: Optional runtime upper bound on the active coordinates.

        Returns:
            Dense ``(lb_full, ub_full)`` of shape ``(#B, dim, 1)``.
        """
        if lb is None and ub is None:
            return (
                self.lb_full.to(dtype=x.dtype, device=x.device),
                self.ub_full.to(dtype=x.dtype, device=x.device),
            )
        lb_act = self.lb if lb is None else lb * self.scale
        ub_act = self.ub if ub is None else ub * self.scale
        lb_act = as_col(lb_act, dtype=x.dtype, device=x.device)
        ub_act = as_col(ub_act, dtype=x.dtype, device=x.device)
        dim = self._dim
        return (
            _dense_bounds(lb_act, self.mask, dim, fill=-math.inf),
            _dense_bounds(ub_act, self.mask, dim, fill=math.inf),
        )

    def project(
        self,
        x: Tensor,
        lb: Tensor | None = None,
        ub: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Clip ``x`` into the box.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            lb: Optional runtime lower bound.
            ub: Optional runtime upper bound.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            The projected point.
        """
        del kwargs
        lb_full, ub_full = self.resolve_bounds(x, lb=lb, ub=ub)
        return box_project(x, lb_full, ub_full)

    def cv(
        self,
        x: Tensor,
        lb: Tensor | None = None,
        ub: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the box violation.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            lb: Optional runtime lower bound.
            ub: Optional runtime upper bound.
            **kwargs: Unused; accepted for the ``Constraint`` interface.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.
        """
        del kwargs
        lb_full, ub_full = self.resolve_bounds(x, lb=lb, ub=ub)
        return box_cv(x, lb_full, ub_full)

    @property
    def dim(self) -> int:
        """Return the primal dimension."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constrained coordinates."""
        return self._n_constraints


def _dense_bounds(bound: Tensor, mask: Tensor, dim: int, fill: float) -> Tensor:
    """Scatter a masked bound into a dense tensor filled with ``fill``.

    Args:
        bound: Bound on the active coordinates, shape ``(B, n_active, 1)``.
        mask: Boolean mask of length ``dim``.
        dim: Full dimension.
        fill: Value used on inactive coordinates.

    Returns:
        Dense bound of shape ``(B, dim, 1)``.
    """
    out = bound.new_full((bound.shape[0], dim, 1), fill)
    scattered = scatter_to_full(bound, mask, dim)
    mask_exp = mask.reshape(1, dim, 1).to(dtype=torch.bool)
    return torch.where(mask_exp, scattered, out)
