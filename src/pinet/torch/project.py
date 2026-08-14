"""Torch projection layer implemented via Douglas-Rachford / ADMM."""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from pinet.constants import Constants
from pinet.torch._ops import equality_project
from pinet.torch._shapes import TensorLike, as_col, as_matrix, restore_cv, restore_x
from pinet.torch.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    CartesianConstraint,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
    SocConstraint,
)
from pinet.torch.constraints.affine_equality import torch_pinv
from pinet.torch.constraints.base import set_buffer
from pinet.torch.dataclasses import EquilibrationParams
from pinet.torch.equilibration import ruiz_equilibration
from pinet.torch.solver.admm import iteration_step, make_admm_loop
from pinet.torch.solver.bicgstab import bicgstab

PROJECTION_DEFAULT_SIGMA = Constants.PROJECTION_DEFAULT_SIGMA
PROJECTION_DEFAULT_OMEGA = Constants.PROJECTION_DEFAULT_OMEGA
PROJECTION_DEFAULT_CHECK_EVERY = Constants.PROJECTION_DEFAULT_CHECK_EVERY
PROJECTION_DEFAULT_TOL = Constants.PROJECTION_DEFAULT_TOL
PROJECTION_DEFAULT_MAX_ITER = Constants.PROJECTION_DEFAULT_MAX_ITER
PROJECTION_DEFAULT_CHECK_REDUCTION = Constants.PROJECTION_DEFAULT_CHECK_REDUCTION


class Project(nn.Module):
    """Differentiable projection layer for convex constraints.

    Construct constraints, then call the module on a batch of points:

    .. code-block:: python

        proj = Project(eq, ineq, box, n_iter=50)
        y = proj(x, eq_b=b)
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None = None,
        ineq_constraint: AffineInequalityConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[NonLinearConstraint] | None = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams | None = None,
        n_iter: int = 100,
        n_iter_bwd: int = 5,
        sigma: float = PROJECTION_DEFAULT_SIGMA,
        omega: float = PROJECTION_DEFAULT_OMEGA,
        fpi: bool = False,
        compile: bool | None = None,
        compile_mode: str | None = None,
        compile_backend: str | None = None,
    ) -> None:
        """Initialize the projection layer.

        Args:
            eq_constraint: Equality constraint.
            ineq_constraint: Affine inequality constraint.
            box_constraint: Box constraint.
            nl_constraints: Non-linear constraints.
            unroll: Differentiate by unrolling ADMM instead of the IFT.
            equilibration_params: Ruiz equilibration settings.
            n_iter: Default number of forward ADMM iterations.
            n_iter_bwd: Default IFT / FPI iterations for the backward pass.
            sigma: Default ADMM stepsize.
            omega: Default relaxation.
            fpi: Use fixed-point iteration instead of BiCGSTAB in the IFT.
            compile: If ``None``, compile on CUDA and stay eager on CPU.
            compile_mode: ``torch.compile`` mode. Defaults to
                ``"reduce-overhead"`` on CUDA and ``"default"`` otherwise.
            compile_backend: Optional ``torch.compile`` backend (for example
                ``"eager"`` to check graph capture without Inductor).
        """
        super().__init__()
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.nl_constraints = nl_constraints
        self.unroll = unroll
        self.equilibration_params = (
            EquilibrationParams()
            if equilibration_params is None
            else equilibration_params
        )
        self.n_iter = n_iter
        self.n_iter_bwd = n_iter_bwd
        self.sigma = sigma
        self.omega = omega
        self.fpi = fpi
        self._compile_flag = compile
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend
        self._loop_cache: dict[int, Callable[..., tuple[Tensor, Tensor]]] = {}
        self._should_compile: bool | None = None
        self.setup()
        self.d_r = self.get_buffer("d_r")
        self.d_c = self.get_buffer("d_c")
        self.scale = self.get_buffer("scale")
        self.eps_t = self.get_buffer("eps_t")

    def setup(self) -> None:
        """Lift constraints, equilibrate, and cache ADMM tensors."""
        constraints = [
            c
            for c in (
                self.eq_constraint,
                self.box_constraint,
                self.ineq_constraint,
                *(self.nl_constraints or []),
            )
            if c is not None
        ]
        assert len(constraints) > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim
        self.is_single_simple_constraint = (
            self.ineq_constraint is None
            and self.nl_constraints is None
            and len(constraints) == 1
        )
        self.dim_lifted = self.dim
        self.n_soc = 0
        self.single_constraint = constraints[0]
        self.lifted_eq_constraint: EqualityConstraint | None = None
        self.lifted_primitive: BoxConstraint | CartesianConstraint | None = None
        dtype, device = _constraint_dtype_device(constraints[0])
        self.d_r = set_buffer(
            self,
            "d_r",
            torch.ones((1, constraints[0].n_constraints, 1), dtype=dtype, device=device),
        )
        self.d_c = set_buffer(
            self, "d_c", torch.ones((1, self.dim, 1), dtype=dtype, device=device)
        )
        self.scale = set_buffer(
            self, "scale", torch.ones((1, self.dim, 1), dtype=dtype, device=device)
        )
        self.eps_t = set_buffer(
            self,
            "eps_t",
            torch.tensor(Constants.SOC_CONSTRAINT_EPSILON, dtype=dtype, device=device),
        )

        if self.is_single_simple_constraint:
            return

        if self.nl_constraints is None:
            if self.ineq_constraint is not None:
                self.dim_lifted += self.ineq_constraint.n_constraints
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            parsed_eq, parsed_primitive, _lift, aux_lift = parser.parse(method=None)
            assert parsed_eq is not None
            assert parsed_primitive is not None
            assert isinstance(parsed_primitive, BoxConstraint)
            if not parsed_eq.var_a_mat and parsed_eq.a_mat.shape[0] == 1:
                scaled_flat, d_r_flat, d_c_flat = ruiz_equilibration(
                    parsed_eq.a_mat[0], self.equilibration_params
                )
                scaled_a_mat = scaled_flat.reshape(1, *parsed_eq.a_mat.shape[1:])
                d_r = d_r_flat.reshape(1, -1, 1)
                d_c = d_c_flat.reshape(1, -1, 1)
            else:
                scaled_a_mat = parsed_eq.a_mat
                n_ineq = (
                    self.ineq_constraint.n_constraints
                    if self.ineq_constraint is not None
                    else 0
                )
                n_eq = (
                    self.eq_constraint.n_constraints
                    if self.eq_constraint is not None
                    else 0
                )
                d_r = torch.ones((1, n_eq + n_ineq, 1), dtype=dtype, device=device)
                d_c = torch.ones((1, self.dim_lifted, 1), dtype=dtype, device=device)
            self.lifted_eq_constraint = EqualityConstraint(
                a_mat=scaled_a_mat,
                b=parsed_eq.b * d_r,
                method="pinv",
                var_b=parsed_eq.var_b,
                var_a_mat=parsed_eq.var_a_mat,
            )
            mask = parsed_primitive.mask.reshape(-1)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            box_scale = 1.0 / d_c[:, idx, :]
            self.lifted_primitive = BoxConstraint(
                lb=parsed_primitive.lb * box_scale,
                ub=parsed_primitive.ub * box_scale,
                mask=parsed_primitive.mask,
                scale=box_scale,
            )
            if aux_lift is not None:
                self.aux_lift = set_buffer(self, "aux_lift", aux_lift)
            self.dim_lifted = int(self.lifted_eq_constraint.a_mat.shape[-1])
            self._set_scaling(d_r, d_c)
        else:
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
                nl_constraints=self.nl_constraints,
            )
            parsed_eq, parsed_primitive, _lift, aux_lift = parser.parse(method="pinv")
            assert parsed_eq is not None
            assert parsed_primitive is not None
            self.lifted_eq_constraint = parsed_eq
            self.lifted_primitive = parsed_primitive
            if aux_lift is not None:
                self.aux_lift = set_buffer(self, "aux_lift", aux_lift)
            self.dim_lifted = int(parsed_eq.a_mat.shape[-1])
            d_r = torch.ones(
                (1, parsed_eq.a_mat.shape[1], 1),
                dtype=parsed_eq.a_mat.dtype,
                device=parsed_eq.a_mat.device,
            )
            d_c = torch.ones(
                (1, self.dim_lifted, 1),
                dtype=parsed_eq.a_mat.dtype,
                device=parsed_eq.a_mat.device,
            )
            self._set_scaling(d_r, d_c)
            if isinstance(parsed_primitive, CartesianConstraint):
                self.n_soc = parsed_primitive.n_nonlinear

    def _set_scaling(self, d_r: Tensor, d_c: Tensor) -> None:
        """Store Ruiz scaling buffers.

        Args:
            d_r: Row scaling, shape ``(1, m, 1)``.
            d_c: Column scaling, shape ``(1, dim_lifted, 1)``.
        """
        self.d_r = d_r
        self.d_c = d_c
        self.scale = d_c[:, : self.dim, :]

    def forward(
        self,
        x: Tensor,
        eq_b: TensorLike | None = None,
        eq_a_mat: TensorLike | None = None,
        box_lb: TensorLike | None = None,
        box_ub: TensorLike | None = None,
        nl_a: Sequence[TensorLike] | None = None,
        nl_b: Sequence[TensorLike] | None = None,
        s0: TensorLike | None = None,
        n_iter: int | None = None,
        n_iter_bwd: int | None = None,
        sigma: float | None = None,
        omega: float | None = None,
        fpi: bool | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Project ``x`` onto the constraint set.

        Args:
            x: Point to project, shape ``(n,)``, ``(B, n)``, or ``(B, n, 1)``.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-cone runtime vector offsets.
            nl_b: Optional per-cone runtime scalar offsets.
            s0: Optional warm-start governing sequence.
            n_iter: Forward iteration override.
            n_iter_bwd: Backward iteration override.
            sigma: ADMM stepsize override.
            omega: Relaxation override.
            fpi: Backward solver override.
            return_state: Also return the governing sequence ``sK``.

        Returns:
            Projected point with the same layout as ``x``, or
            ``(projected_point, sK)`` when ``return_state`` is true.
        """
        x_in = x if isinstance(x, Tensor) else as_col(x)
        x_col = as_col(x, dtype=x_in.dtype if isinstance(x, Tensor) else None)
        n_iter_use = self.n_iter if n_iter is None else n_iter
        n_iter_bwd_use = self.n_iter_bwd if n_iter_bwd is None else n_iter_bwd
        sigma_use = self.sigma if sigma is None else sigma
        omega_use = self.omega if omega is None else omega
        fpi_use = self.fpi if fpi is None else fpi
        assert n_iter_use > 0, "Number of iterations must be positive."

        if self.is_single_simple_constraint:
            y_col = self._project_single(
                x_col, eq_b=eq_b, eq_a_mat=eq_a_mat, box_lb=box_lb, box_ub=box_ub
            )
            y_out = restore_x(y_col, x_in)
            if return_state:
                return y_out, x_col
            return y_out

        tensors = self._prepare_admm_tensors(
            x_col,
            eq_b=eq_b,
            eq_a_mat=eq_a_mat,
            box_lb=box_lb,
            box_ub=box_ub,
            nl_a=nl_a,
            nl_b=nl_b,
            s0=s0,
            sigma=sigma_use,
            omega=omega_use,
        )
        if self.unroll:
            loop = self._get_loop(n_iter_use, x_col.device)
            y_col, s_k = loop(*tensors)
        else:
            y_col, s_k = _project_implicit(
                n_iter_bwd=n_iter_bwd_use,
                fpi=fpi_use,
                dim=self.dim,
                n_soc=self.n_soc,
                loop=self._get_loop(n_iter_use, x_col.device),
                tensors=tensors,
            )
        y_out = restore_x(y_col, x_in)
        if return_state:
            return y_out, s_k
        return y_out

    def cv(
        self,
        y: Tensor,
        eq_b: TensorLike | None = None,
        eq_a_mat: TensorLike | None = None,
        box_lb: TensorLike | None = None,
        box_ub: TensorLike | None = None,
        nl_a: Sequence[TensorLike] | None = None,
        nl_b: Sequence[TensorLike] | None = None,
    ) -> Tensor:
        """Maximum constraint violation of a (possibly unlifted) point.

        Args:
            y: Point to evaluate.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-cone runtime vector offsets.
            nl_b: Optional per-cone runtime scalar offsets.

        Returns:
            Violation of shape ``(B,)`` or ``(B, 1, 1)`` matching ``y``'s rank.
        """
        y_in = y if isinstance(y, Tensor) else as_col(y)
        y_col = as_col(y)
        if self.is_single_simple_constraint:
            cv_val = self._cv_single(
                y_col, eq_b=eq_b, eq_a_mat=eq_a_mat, box_lb=box_lb, box_ub=box_ub
            )
            return restore_cv(cv_val, y_in)
        y_lifted = y_col
        if y_col.shape[1] != self.dim_lifted:
            y_lifted = self._lift_x(y_col)
        b_use, a_use, _ = self._resolve_equality(y_col, eq_b=eq_b, eq_a_mat=eq_a_mat)
        assert self.lifted_eq_constraint is not None
        assert self.lifted_primitive is not None
        eq_cv = self.lifted_eq_constraint.cv(y_lifted, b=b_use, a_mat=a_use)
        prim_cv = self._primitive_cv(
            y_lifted, box_lb=box_lb, box_ub=box_ub, nl_a=nl_a, nl_b=nl_b
        )
        return restore_cv(torch.maximum(eq_cv, prim_cv), y_in)

    def call_and_check(
        self,
        sigma: float = PROJECTION_DEFAULT_SIGMA,
        omega: float = PROJECTION_DEFAULT_OMEGA,
        check_every: int = PROJECTION_DEFAULT_CHECK_EVERY,
        tol: float = PROJECTION_DEFAULT_TOL,
        max_iter: int = PROJECTION_DEFAULT_MAX_ITER,
        reduction: str | float = PROJECTION_DEFAULT_CHECK_REDUCTION,
    ) -> Callable[..., tuple[Tensor, Tensor, int]]:
        """Return a function that projects until the violation is small.

        Args:
            sigma: ADMM parameter.
            omega: ADMM parameter.
            check_every: Iterations between constraint checks.
            tol: Constraint-violation tolerance.
            max_iter: Maximum number of iterations.
            reduction: ``"max"``, ``"mean"``, or a fraction in ``(0, 1)``.

        Returns:
            Callable mapping ``x`` (and optional runtime constraint data) to
            ``(projected_point, terminated, iterations)``.
        """

        def _check(y: Tensor, x_in: Tensor, **cv_kwargs: Any) -> bool:
            del x_in
            cv_val = self.cv(y, **cv_kwargs)
            if reduction == "max":
                return bool(cv_val.max() < tol)
            if reduction == "mean":
                return bool(cv_val.mean() < tol)
            if isinstance(reduction, float) and 0 < reduction < 1:
                return bool((cv_val < tol).to(dtype=cv_val.dtype).mean() >= reduction)
            raise ValueError(
                f"Invalid reduction method {reduction}. "
                "Valid options are: 'max', 'mean', or a float in (0, 1)."
            )

        def project_and_check(
            x: Tensor,
            **kwargs: Any,
        ) -> tuple[Tensor, Tensor, int]:
            iter_exec = 0
            terminated = False
            s0 = kwargs.pop("s0", None)
            y_proj = x
            s_k = s0
            while not (terminated or iter_exec >= max_iter):
                result = self.forward(
                    x,
                    s0=s_k,
                    n_iter=check_every,
                    sigma=sigma,
                    omega=omega,
                    return_state=True,
                    **kwargs,
                )
                y_proj, s_k = cast(tuple[Tensor, Tensor], result)
                iter_exec += check_every
                terminated = _check(y_proj, x, **kwargs)
            return y_proj, x.new_tensor(terminated), iter_exec

        return project_and_check

    def _project_single(
        self,
        x: Tensor,
        eq_b: TensorLike | None,
        eq_a_mat: TensorLike | None,
        box_lb: TensorLike | None,
        box_ub: TensorLike | None,
    ) -> Tensor:
        """Closed-form projection for a single equality or box.

        Args:
            x: Point to project, shape ``(B, n, 1)``.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.

        Returns:
            Projected point.
        """
        constraint = self.single_constraint
        if isinstance(constraint, EqualityConstraint):
            b = None if eq_b is None else as_col(eq_b, dtype=x.dtype, device=x.device)
            a_mat = (
                None
                if eq_a_mat is None
                else as_matrix(eq_a_mat, dtype=x.dtype, device=x.device)
            )
            a_pinv = None if a_mat is None else torch_pinv(a_mat)
            return constraint.project(x, b=b, a_mat=a_mat, a_mat_pinv=a_pinv)
        assert isinstance(constraint, BoxConstraint)
        lb = None if box_lb is None else as_col(box_lb, dtype=x.dtype, device=x.device)
        ub = None if box_ub is None else as_col(box_ub, dtype=x.dtype, device=x.device)
        return constraint.project(x, lb=lb, ub=ub)

    def _cv_single(
        self,
        x: Tensor,
        eq_b: TensorLike | None,
        eq_a_mat: TensorLike | None,
        box_lb: TensorLike | None,
        box_ub: TensorLike | None,
    ) -> Tensor:
        """Constraint violation for a single simple constraint.

        Args:
            x: Point to evaluate.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.

        Returns:
            Violation of shape ``(B, 1, 1)``.
        """
        constraint = self.single_constraint
        if isinstance(constraint, EqualityConstraint):
            b = None if eq_b is None else as_col(eq_b, dtype=x.dtype, device=x.device)
            a_mat = (
                None
                if eq_a_mat is None
                else as_matrix(eq_a_mat, dtype=x.dtype, device=x.device)
            )
            return constraint.cv(x, b=b, a_mat=a_mat)
        assert isinstance(constraint, BoxConstraint)
        lb = None if box_lb is None else as_col(box_lb, dtype=x.dtype, device=x.device)
        ub = None if box_ub is None else as_col(box_ub, dtype=x.dtype, device=x.device)
        return constraint.cv(x, lb=lb, ub=ub)

    def _lift_x(self, x: Tensor) -> Tensor:
        """Lift a primal point by appending auxiliary slacks.

        Args:
            x: Unlifted point, shape ``(B, dim, 1)``.

        Returns:
            Lifted point, shape ``(B, dim_lifted, 1)``.
        """
        aux_lift = getattr(self, "aux_lift", None)
        if aux_lift is None:
            return x
        aux = aux_lift.to(dtype=x.dtype, device=x.device) @ x
        return torch.cat([x, aux], dim=1)

    def _resolve_equality(
        self,
        x: Tensor,
        eq_b: TensorLike | None,
        eq_a_mat: TensorLike | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Resolve lifted equality parameters for the current call.

        Args:
            x: Unlifted point, used for dtype/device and batch size.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.

        Returns:
            Tuple ``(b, a_mat, a_pinv)`` in the lifted space.
        """
        assert self.lifted_eq_constraint is not None
        lifted = self.lifted_eq_constraint
        if eq_a_mat is not None:
            a_raw = as_matrix(eq_a_mat, dtype=x.dtype, device=x.device)
            b_raw = (
                lifted.b if eq_b is None else as_col(eq_b, dtype=x.dtype, device=x.device)
            )
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(
                    a_mat=a_raw, b=b_raw, method="pinv", var_b=True, var_a_mat=True
                ),
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
                nl_constraints=self.nl_constraints,
            )
            parsed_eq, _, _, _ = parser.parse(method="pinv")
            assert parsed_eq is not None
            a_mat = parsed_eq.a_mat
            a_pinv = parsed_eq.a_mat_pinv
            assert a_pinv is not None
        else:
            a_mat = lifted.a_mat.to(dtype=x.dtype, device=x.device)
            assert lifted.a_mat_pinv is not None
            a_pinv = lifted.a_mat_pinv.to(dtype=x.dtype, device=x.device)

        if eq_b is not None:
            b_orig = as_col(eq_b, dtype=x.dtype, device=x.device)
            n_pad = self.dim_lifted - self.dim
            if n_pad > 0:
                b_orig = torch.cat(
                    [b_orig, b_orig.new_zeros(b_orig.shape[0], n_pad, 1)], dim=1
                )
            b_use = b_orig * self.d_r.to(dtype=x.dtype, device=x.device)
        else:
            b_use = lifted.b.to(dtype=x.dtype, device=x.device)
        return b_use, a_mat, a_pinv

    def _prepare_admm_tensors(
        self,
        x: Tensor,
        eq_b: TensorLike | None,
        eq_a_mat: TensorLike | None,
        box_lb: TensorLike | None,
        box_ub: TensorLike | None,
        nl_a: Sequence[TensorLike] | None,
        nl_b: Sequence[TensorLike] | None,
        s0: TensorLike | None,
        sigma: float,
        omega: float,
    ) -> tuple[Tensor, ...]:
        """Pack tensor arguments for the compiled ADMM loop.

        Args:
            x: Unlifted point, shape ``(B, dim, 1)``.
            eq_b: Optional runtime equality right-hand side.
            eq_a_mat: Optional runtime equality matrix.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-cone runtime vector offsets.
            nl_b: Optional per-cone runtime scalar offsets.
            s0: Optional warm-start governing sequence.
            sigma: ADMM stepsize.
            omega: Relaxation.

        Returns:
            Positional tensor tuple consumed by ``make_admm_loop``.
        """
        b_use, a_mat, a_pinv = self._resolve_equality(x, eq_b=eq_b, eq_a_mat=eq_a_mat)
        lb, ub = self._resolve_box(x, box_lb=box_lb, box_ub=box_ub)
        soc_pack, eps = self._resolve_soc(x, nl_a=nl_a, nl_b=nl_b)
        if s0 is None:
            s0_t = x.new_zeros(x.shape[0], self.dim_lifted, 1)
        else:
            s0_t = as_col(s0, dtype=x.dtype, device=x.device)
        sigma_t = x.new_tensor(float(sigma))
        omega_t = x.new_tensor(float(omega))
        scale = self.scale.to(dtype=x.dtype, device=x.device)
        d_c = self.d_c[:, : self.dim, :].to(dtype=x.dtype, device=x.device)
        return (
            s0_t,
            x,
            a_mat,
            a_pinv,
            b_use,
            lb,
            ub,
            scale,
            d_c,
            sigma_t,
            omega_t,
            eps,
            *soc_pack,
        )

    def _resolve_box(
        self,
        x: Tensor,
        box_lb: TensorLike | None,
        box_ub: TensorLike | None,
    ) -> tuple[Tensor, Tensor]:
        """Resolve dense box bounds in the lifted space.

        Args:
            x: Unlifted point, used for dtype/device.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.

        Returns:
            Dense ``(lb, ub)`` of shape ``(#B, dim_lifted, 1)``.
        """
        primitive = self.lifted_primitive
        lb_t = None if box_lb is None else as_col(box_lb, dtype=x.dtype, device=x.device)
        ub_t = None if box_ub is None else as_col(box_ub, dtype=x.dtype, device=x.device)
        if isinstance(primitive, BoxConstraint):
            return primitive.resolve_bounds(x, lb=lb_t, ub=ub_t)
        assert isinstance(primitive, CartesianConstraint)
        if primitive.box_constraint is not None:
            return primitive.box_constraint.resolve_bounds(x, lb=lb_t, ub=ub_t)
        inf = x.new_tensor(float("inf"))
        lb = inf.new_full((1, self.dim_lifted, 1), float("-inf"))
        ub = inf.new_full((1, self.dim_lifted, 1), float("inf"))
        return lb, ub

    def _resolve_soc(
        self,
        x: Tensor,
        nl_a: Sequence[TensorLike] | None,
        nl_b: Sequence[TensorLike] | None,
    ) -> tuple[list[Tensor], Tensor]:
        """Resolve SOC tensors for the ADMM primitive projection.

        Args:
            x: Unlifted point, used for dtype/device.
            nl_a: Optional per-cone runtime vector offsets.
            nl_b: Optional per-cone runtime scalar offsets.

        Returns:
            Flattened SOC pack and the stabilizer ``eps``.
        """
        eps = self.eps_t.to(dtype=x.dtype, device=x.device)
        primitive = self.lifted_primitive
        if not isinstance(primitive, CartesianConstraint) or self.n_soc == 0:
            return [], eps
        pack: list[Tensor] = []
        for i, soc in enumerate(primitive.nl_constraints):
            assert isinstance(soc, SocConstraint)
            a_i = (
                None if nl_a is None else as_col(nl_a[i], dtype=x.dtype, device=x.device)
            )
            b_i = (
                None if nl_b is None else as_col(nl_b[i], dtype=x.dtype, device=x.device)
            )
            a_full, b_full, eps = soc.resolve_offsets(x, a=a_i, b=b_i)
            pack.extend(
                [
                    soc.mask_u.to(device=x.device),
                    soc.mask_t.to(device=x.device),
                    a_full,
                    b_full,
                ]
            )
        return pack, eps

    def _primitive_cv(
        self,
        y_lifted: Tensor,
        box_lb: TensorLike | None,
        box_ub: TensorLike | None,
        nl_a: Sequence[TensorLike] | None,
        nl_b: Sequence[TensorLike] | None,
    ) -> Tensor:
        """Constraint violation of the lifted primitive.

        Args:
            y_lifted: Lifted point.
            box_lb: Optional runtime box lower bound.
            box_ub: Optional runtime box upper bound.
            nl_a: Optional per-cone runtime vector offsets.
            nl_b: Optional per-cone runtime scalar offsets.

        Returns:
            Violation of shape ``(B, 1, 1)``.
        """
        primitive = self.lifted_primitive
        assert primitive is not None
        lb_t = (
            None
            if box_lb is None
            else as_col(box_lb, dtype=y_lifted.dtype, device=y_lifted.device)
        )
        ub_t = (
            None
            if box_ub is None
            else as_col(box_ub, dtype=y_lifted.dtype, device=y_lifted.device)
        )
        if isinstance(primitive, BoxConstraint):
            return primitive.cv(y_lifted, lb=lb_t, ub=ub_t)
        assert isinstance(primitive, CartesianConstraint)
        a_list = None
        b_list = None
        if nl_a is not None:
            a_list = [
                as_col(a, dtype=y_lifted.dtype, device=y_lifted.device) for a in nl_a
            ]
        if nl_b is not None:
            b_list = [
                as_col(b, dtype=y_lifted.dtype, device=y_lifted.device) for b in nl_b
            ]
        return primitive.cv(y_lifted, box_lb=lb_t, box_ub=ub_t, nl_a=a_list, nl_b=b_list)

    def _get_loop(
        self, n_iter: int, device: torch.device
    ) -> Callable[..., tuple[Tensor, Tensor]]:
        """Return a (possibly compiled) ADMM loop specialized to ``n_iter``.

        Args:
            n_iter: Number of forward iterations.
            device: Device of the current batch, used to decide compilation.

        Returns:
            ADMM loop callable.
        """
        if n_iter in self._loop_cache:
            return self._loop_cache[n_iter]
        loop = make_admm_loop(n_iter, self.dim, self.n_soc)
        if self._should_compile is None:
            if self._compile_flag is None:
                self._should_compile = device.type == "cuda"
            else:
                self._should_compile = self._compile_flag
        if self._should_compile:
            mode = self._compile_mode
            if mode is None:
                mode = "reduce-overhead" if device.type == "cuda" else "default"
            compiled = (
                torch.compile(
                    loop, dynamic=False, mode=mode, backend=self._compile_backend
                )
                if self._compile_backend is not None
                else torch.compile(loop, dynamic=False, mode=mode)
            )
            raw = loop

            def _loop_with_fallback(
                *args: Tensor,
            ) -> tuple[Tensor, Tensor]:
                """Run the compiled loop, falling back to eager on failure.

                Args:
                    *args: ADMM loop tensors.

                Returns:
                    Pair ``(projected_point, governing_sequence)``.
                """
                try:
                    return compiled(*args)
                except (RuntimeError, OSError, ValueError, TypeError) as exc:
                    warnings.warn(
                        f"torch.compile failed ({type(exc).__name__}: {exc}); "
                        "falling back to eager ADMM.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._should_compile = False
                    self._loop_cache[n_iter] = raw
                    return raw(*args)

            loop = _loop_with_fallback
        self._loop_cache[n_iter] = loop
        return loop


def _project_implicit(
    n_iter_bwd: int,
    fpi: bool,
    dim: int,
    n_soc: int,
    loop: Callable[..., tuple[Tensor, Tensor]],
    tensors: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    """Run ADMM with an implicit-function backward.

    Args:
        n_iter_bwd: IFT / FPI iterations.
        fpi: Whether to use fixed-point iteration.
        dim: Original primal dimension.
        n_soc: Number of SOC cones.
        loop: Compiled or eager ADMM loop.
        tensors: Inputs of ``loop``, starting with ``(s0, x, ...)``.

    Returns:
        Pair ``(projected_point, governing_sequence)``.
    """

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *inputs: Tensor) -> tuple[Tensor, Tensor]:
            """Run the ADMM loop and stash residuals.

            Args:
                ctx: Autograd context.
                *inputs: ADMM loop tensors.

            Returns:
                Pair ``(projected_point, governing_sequence)``.
            """
            y_out, s_k = loop(*inputs)
            ctx.n_iter_bwd = n_iter_bwd
            ctx.fpi = fpi
            ctx.dim = dim
            ctx.n_soc = n_soc
            ctx.s_k = s_k
            ctx.save_for_backward(*inputs)
            return y_out, s_k

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            """Apply the implicit-function VJP.

            Args:
                ctx: Autograd context from the forward pass.
                *grad_outputs: Incoming cotangents of ``(y, sK)``.

            Returns:
                Cotangents matching the forward tensor arguments.
            """
            grad_y = grad_outputs[0] if len(grad_outputs) > 0 else None
            saved = ctx.saved_tensors
            (
                _s0,
                x,
                a_mat,
                a_pinv,
                b,
                lb,
                ub,
                scale,
                d_c,
                sigma,
                omega,
                eps,
                *soc_pack,
            ) = saved
            s_k = ctx.s_k
            if grad_y is None:
                grad_y = torch.zeros_like(x)
            cot = grad_y * d_c
            zeros_aux = cot.new_zeros(cot.shape[0], s_k.shape[1] - dim, 1)
            cot_lifted = torch.cat([cot, zeros_aux], dim=1)
            cot_eq = equality_project(cot_lifted, a_mat, a_pinv, torch.zeros_like(b))

            def step_s(state: Tensor) -> Tensor:
                return iteration_step(
                    state,
                    x,
                    a_mat,
                    a_pinv,
                    b,
                    lb,
                    ub,
                    scale,
                    sigma,
                    omega,
                    dim,
                    n_soc,
                    eps,
                    soc_pack,
                )

            def step_y(point: Tensor) -> Tensor:
                return iteration_step(
                    s_k,
                    point,
                    a_mat,
                    a_pinv,
                    b,
                    lb,
                    ub,
                    scale,
                    sigma,
                    omega,
                    dim,
                    n_soc,
                    eps,
                    soc_pack,
                )

            vjp_s = torch.func.vjp(step_s, s_k)[1]
            vjp_y = torch.func.vjp(step_y, x)[1]
            if fpi:
                lam = torch.zeros_like(s_k)
                for _ in range(ctx.n_iter_bwd):
                    lam = vjp_s(lam)[0] + cot_eq
            else:

                def _op(vec: Tensor) -> Tensor:
                    return vec - vjp_s(vec)[0]

                lam = bicgstab(_op, cot_eq, maxiter=ctx.n_iter_bwd)
            grad_x = vjp_y(lam)[0]
            grads: list[Tensor | None] = [None] * len(saved)
            grads[1] = grad_x
            return tuple(grads)

    return cast(tuple[Tensor, Tensor], _Fn.apply(*tensors))


def _constraint_dtype_device(constraint: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Infer dtype and device from a constraint module.

    Args:
        constraint: Constraint whose buffers are inspected.

    Returns:
        Pair ``(dtype, device)``.
    """
    for tensor in constraint.buffers():
        if tensor.is_floating_point():
            return tensor.dtype, tensor.device
    for tensor in constraint.buffers():
        return tensor.dtype, tensor.device
    return torch.float32, torch.device("cpu")
