"""Shape and dtype helpers for the Torch projection API."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

TensorLike = Tensor | np.ndarray[Any, np.dtype[Any]]


def as_tensor(
    data: TensorLike | float | bool,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Convert ``data`` to a tensor, preserving floating dtypes when possible.

    Args:
        data: Array, tensor, or scalar to convert.
        dtype: Optional override dtype.
        device: Optional override device.

    Returns:
        A tensor on the requested device.
    """
    if isinstance(data, Tensor):
        tensor = data
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        return tensor

    array = np.asarray(data)
    if dtype is None:
        if array.dtype in (np.bool_, bool):
            dtype = torch.bool
        elif array.dtype == np.float64:
            dtype = torch.float64
        elif array.dtype == np.float32:
            dtype = torch.float32
        elif np.issubdtype(array.dtype, np.integer):
            dtype = torch.int64 if array.dtype == np.int64 else None
    return torch.as_tensor(array, dtype=dtype, device=device)


def as_col(
    data: TensorLike,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Coerce a vector to column layout ``(B, n, 1)``.

    Args:
        data: Vector of shape ``(n,)``, ``(B, n)``, or ``(B, n, 1)``.
        dtype: Optional override dtype.
        device: Optional override device.

    Returns:
        Column tensor of shape ``(B, n, 1)``.

    Raises:
        ValueError: If ``data`` does not have 1, 2, or 3 dimensions.
    """
    tensor = as_tensor(data, dtype=dtype, device=device)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1, 1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 3:
        if tensor.shape[-1] != 1:
            raise ValueError(
                f"3D vectors must have shape (batch, dim, 1), got {tuple(tensor.shape)}."
            )
        return tensor
    raise ValueError(
        f"Expected a vector with 1, 2, or 3 dimensions, got shape {tuple(tensor.shape)}."
    )


def as_matrix(
    data: TensorLike,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Coerce a matrix to batched layout ``(B, m, n)``.

    Args:
        data: Matrix of shape ``(m, n)`` or ``(B, m, n)``.
        dtype: Optional override dtype.
        device: Optional override device.

    Returns:
        Batched matrix of shape ``(B, m, n)``.

    Raises:
        ValueError: If ``data`` does not have 2 or 3 dimensions.
    """
    tensor = as_tensor(data, dtype=dtype, device=device)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(
        f"Expected a matrix with 2 or 3 dimensions, got shape {tuple(tensor.shape)}."
    )


def restore_x(y_col: Tensor, x_in: Tensor) -> Tensor:
    """Restore ``y_col`` to the rank of the user-provided ``x_in``.

    Args:
        y_col: Projected point in column layout ``(B, n, 1)``.
        x_in: Original user tensor.

    Returns:
        Projected point with the same number of dimensions as ``x_in``.
    """
    if x_in.ndim == 1:
        return y_col.reshape(-1)
    if x_in.ndim == 2:
        return y_col.squeeze(-1)
    return y_col


def restore_cv(cv_col: Tensor, x_in: Tensor) -> Tensor:
    """Restore a constraint-violation tensor to a torch-friendly rank.

    Args:
        cv_col: Violation in layout ``(B, 1, 1)``.
        x_in: Original user tensor used to choose the output rank.

    Returns:
        ``(B,)`` when ``x_in`` is 1D/2D, otherwise ``(B, 1, 1)``.
    """
    if x_in.ndim <= 2:
        return cv_col.reshape(-1)
    return cv_col


def scatter_to_full(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Scatter masked values into a dense ``(B, dim, 1)`` tensor.

    Args:
        values: Values on the active coordinates, shape ``(B, n_active, 1)``.
        mask: Boolean mask of length ``dim``.
        dim: Full primal dimension.

    Returns:
        Dense tensor with ``values`` on ``mask`` and zeros elsewhere.
    """
    out = values.new_zeros(values.shape[0], dim, 1)
    mask_bool = mask.reshape(-1).to(dtype=torch.bool)
    idx = mask_bool.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return out
    out = out.clone()
    out[:, idx, :] = values
    return out
