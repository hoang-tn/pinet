"""Abstract constraint interface for the Torch backend."""

from typing import Any

from torch import Tensor, nn


def set_buffer(module: nn.Module, name: str, tensor: Tensor) -> Tensor:
    """Register ``tensor`` as a buffer and return it for typed assignment.

    Args:
        module: Module that owns the buffer.
        name: Buffer name.
        tensor: Value to store.

    Returns:
        The registered tensor.
    """
    module.register_buffer(name, tensor)
    return tensor


class Constraint(nn.Module):
    """Abstract constraint set.

    Subclasses must override ``project``, ``cv``, ``dim`` and ``n_constraints``.
    """

    def project(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Project the input onto the feasible set.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            **kwargs: Optional runtime constraint parameters.

        Returns:
            The projected point.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.project must be implemented by a subclass."
        )

    def cv(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Compute the constraint violation.

        Args:
            x: Point to evaluate, shape ``(B, n, 1)``.
            **kwargs: Optional runtime constraint parameters.

        Returns:
            Constraint violation of shape ``(B, 1, 1)``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.cv must be implemented by a subclass."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.dim must be implemented by a subclass."
        )

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.n_constraints must be implemented by a subclass."
        )
