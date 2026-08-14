"""Parser of constraints into a lifted representation (Torch)."""

from collections.abc import Callable

import torch
from torch import Tensor

from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint
from .cartesian_constraint import CartesianConstraint
from .non_linear import NonLinearConstraint
from .soc_constraint import SocConstraint

ParseLifter = Callable[[Tensor], Tensor]
ParseResult = tuple[
    EqualityConstraint | None,
    BoxConstraint | CartesianConstraint | None,
    ParseLifter,
    Tensor | None,
]


class ConstraintParser:
    """Parse constraints into a lifted equality plus a primitive set."""

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None = None,
        ineq_constraint: AffineInequalityConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[NonLinearConstraint] | None = None,
    ) -> None:
        """Initialize the constraint parser.

        Args:
            eq_constraint: An equality constraint.
            ineq_constraint: An inequality constraint.
            box_constraint: A box constraint.
            nl_constraints: Non-linear constraints.
        """
        self.ineq_constraint = ineq_constraint
        self.nl_constraints = nl_constraints
        self._identity_mode = ineq_constraint is None and nl_constraints is None

        if self._identity_mode:
            self.eq_constraint = eq_constraint
            self.box_constraint = box_constraint
            self.dim = 0
            self.n_eq = 0
            self.n_ineq = 0
            return

        if ineq_constraint is not None:
            self.dim = ineq_constraint.dim
        else:
            assert nl_constraints is not None
            self.dim = nl_constraints[0].dim

        if eq_constraint is None:
            if ineq_constraint is not None:
                dtype = ineq_constraint.c_mat.dtype
                device = ineq_constraint.c_mat.device
            else:
                assert nl_constraints is not None
                dtype = nl_constraints[0].a_mat.dtype
                device = nl_constraints[0].a_mat.device
            eq_constraint = EqualityConstraint(
                a_mat=torch.empty((1, 0, self.dim), dtype=dtype, device=device),
                b=torch.empty((1, 0, 1), dtype=dtype, device=device),
                method=None,
                var_b=False,
                var_a_mat=False,
            )

        self.eq_constraint = eq_constraint
        self.n_eq = eq_constraint.n_constraints
        self.n_ineq = ineq_constraint.n_constraints if ineq_constraint is not None else 0
        self.box_constraint = box_constraint

        if ineq_constraint is not None:
            assert (
                self.eq_constraint.a_mat.shape[0] == ineq_constraint.c_mat.shape[0]
                or self.eq_constraint.a_mat.shape[0] == 1
                or ineq_constraint.c_mat.shape[0] == 1
            ), "Batch sizes of a_mat and c_mat must be consistent."
        if nl_constraints is not None:
            for non_linear in nl_constraints:
                assert non_linear.a_mat.shape[0] == 1, (
                    "Batch size of non-linear constraint a_mat must be 1."
                )
                if non_linear.f is not None:
                    assert non_linear.f.shape[0] == 1, (
                        "Batch size of non-linear constraint f must be 1."
                    )
            if ineq_constraint is not None:
                assert ineq_constraint.c_mat.shape[0] == 1, (
                    "Batch size of inequality constraint must be 1."
                )

    def parse(self, method: str | None = "pinv") -> ParseResult:
        """Parse the constraints into a lifted representation.

        Args:
            method: Method used to solve linear systems.

        Returns:
            Tuple ``(eq, primitive, lift_fn, aux_lift_mat)``.
        """
        if self._identity_mode:
            return (self.eq_constraint, self.box_constraint, lambda y: y, None)
        if self.nl_constraints is not None:
            return self.parse_non_linear(method)
        return self.parse_polytope(method)

    def parse_polytope(
        self, method: str | None = "pinv"
    ) -> tuple[EqualityConstraint, BoxConstraint, ParseLifter, Tensor]:
        """Parse equality + inequality + box constraints into lifted form.

        Args:
            method: Method used to solve linear systems.

        Returns:
            Lifted equality, lifted box, lift function, and slack map ``C``.
        """
        ineq_constraint = self.ineq_constraint
        assert ineq_constraint is not None
        eq_constraint = self.eq_constraint
        assert eq_constraint is not None

        mb_ac = max(eq_constraint.a_mat.shape[0], ineq_constraint.c_mat.shape[0])
        first_row = torch.cat(
            [
                eq_constraint.a_mat,
                eq_constraint.a_mat.new_zeros(
                    eq_constraint.a_mat.shape[0], self.n_eq, self.n_ineq
                ),
            ],
            dim=2,
        )
        first_row_batched = first_row.repeat(mb_ac // eq_constraint.a_mat.shape[0], 1, 1)
        eye = torch.eye(
            self.n_ineq,
            dtype=ineq_constraint.c_mat.dtype,
            device=ineq_constraint.c_mat.device,
        ).reshape(1, self.n_ineq, self.n_ineq)
        second_row = torch.cat(
            [
                ineq_constraint.c_mat,
                -eye.repeat(ineq_constraint.c_mat.shape[0], 1, 1),
            ],
            dim=2,
        )
        second_row_batched = second_row.repeat(
            mb_ac // ineq_constraint.c_mat.shape[0], 1, 1
        )
        a_mat_lifted = torch.cat([first_row_batched, second_row_batched], dim=1)
        b_lifted = torch.cat(
            [
                eq_constraint.b,
                eq_constraint.b.new_zeros(eq_constraint.b.shape[0], self.n_ineq, 1),
            ],
            dim=1,
        )
        eq_lifted = EqualityConstraint(
            a_mat=a_mat_lifted,
            b=b_lifted,
            method=method,
            var_b=eq_constraint.var_b,
            var_a_mat=eq_constraint.var_a_mat,
        )

        if self.box_constraint is None:
            mask_box = torch.cat(
                [
                    torch.zeros(self.dim, dtype=torch.bool),
                    torch.ones(self.n_ineq, dtype=torch.bool),
                ]
            )
            box_lifted = BoxConstraint(
                lb=ineq_constraint.lb, ub=ineq_constraint.ub, mask=mask_box
            )
        else:
            mask_box = torch.cat(
                [
                    self.box_constraint.mask.reshape(-1).to(dtype=torch.bool),
                    torch.ones(
                        self.n_ineq,
                        dtype=torch.bool,
                        device=self.box_constraint.mask.device,
                    ),
                ]
            )
            mblb = max(self.box_constraint.lb.shape[0], ineq_constraint.lb.shape[0])
            lifted_lb = torch.cat(
                [
                    self.box_constraint.lb.repeat(
                        mblb // self.box_constraint.lb.shape[0], 1, 1
                    ),
                    ineq_constraint.lb.repeat(mblb // ineq_constraint.lb.shape[0], 1, 1),
                ],
                dim=1,
            )
            mbub = max(self.box_constraint.ub.shape[0], ineq_constraint.ub.shape[0])
            lifted_ub = torch.cat(
                [
                    self.box_constraint.ub.repeat(
                        mbub // self.box_constraint.ub.shape[0], 1, 1
                    ),
                    ineq_constraint.ub.repeat(mbub // ineq_constraint.ub.shape[0], 1, 1),
                ],
                dim=1,
            )
            box_lifted = BoxConstraint(lb=lifted_lb, ub=lifted_ub, mask=mask_box)

        c_mat = ineq_constraint.c_mat

        def lift(y: Tensor) -> Tensor:
            return torch.cat([y, c_mat @ y], dim=1)

        return (eq_lifted, box_lifted, lift, c_mat)

    def parse_non_linear(
        self, method: str | None = "pinv"
    ) -> tuple[EqualityConstraint, CartesianConstraint, ParseLifter, Tensor]:
        """Parse equality + inequality + box + non-linear constraints.

        Args:
            method: Method used to solve linear systems.

        Returns:
            Lifted equality, cartesian primitive, lift function, and aux map.
        """
        nl_constraints = self.nl_constraints
        assert nl_constraints is not None
        eq_constraint = self.eq_constraint
        assert eq_constraint is not None

        all_matrices: list[Tensor] = [eq_constraint.a_mat]
        dims: list[int] = [eq_constraint.dim]
        if self.ineq_constraint is not None:
            all_matrices.append(self.ineq_constraint.c_mat)
            dims.append(self.ineq_constraint.n_constraints)
        for non_linear in nl_constraints:
            all_matrices.append(non_linear.a_mat)
            dims.append(int(non_linear.a_mat.shape[1]))
            f_row = (
                non_linear.f
                if non_linear.f is not None
                else non_linear.a_mat.new_zeros(1, 1, non_linear.a_mat.shape[2])
            )
            all_matrices.append(f_row)
            dims[-1] += 1

        n_aux = sum(dims[1:])
        n_tot = sum(dims)
        max_batch = max(matrix.shape[0] for matrix in all_matrices)
        tiled = [
            matrix.repeat(max_batch // matrix.shape[0], 1, 1) for matrix in all_matrices
        ]
        lifted_a_b1 = torch.cat(tiled, dim=1)
        lifted_a_b2 = lifted_a_b1.new_zeros(max_batch, lifted_a_b1.shape[1], n_aux)
        a_lifted = torch.cat([lifted_a_b1, lifted_a_b2], dim=2)
        start_row = eq_constraint.n_constraints
        start_col = eq_constraint.dim
        eye_aux = torch.eye(n_aux, dtype=a_lifted.dtype, device=a_lifted.device).reshape(
            1, n_aux, n_aux
        )
        a_lifted = a_lifted.clone()
        a_lifted[:, start_row:, start_col:] = (
            a_lifted[:, start_row:, start_col:] - eye_aux
        )
        b_lifted = torch.cat(
            [
                eq_constraint.b,
                eq_constraint.b.new_zeros(eq_constraint.b.shape[0], n_aux, 1),
            ],
            dim=1,
        )
        eq_lifted = EqualityConstraint(
            a_mat=a_lifted,
            b=b_lifted,
            method=method,
            var_b=eq_constraint.var_b,
            var_a_mat=eq_constraint.var_a_mat,
        )

        prim_constraints: list[SocConstraint] = []
        n_curr = eq_constraint.dim
        box_lifted: BoxConstraint | None = None
        if self.box_constraint is not None or self.ineq_constraint is not None:
            if self.box_constraint is not None:
                mask_box_init = self.box_constraint.mask.reshape(-1).to(dtype=torch.bool)
                box_lb_init = self.box_constraint.lb
                box_ub_init = self.box_constraint.ub
            else:
                mask_box_init = torch.zeros(self.dim, dtype=torch.bool)
                box_lb_init = torch.full((1, 0, 1), float("-inf"))
                box_ub_init = torch.full((1, 0, 1), float("inf"))
            if self.ineq_constraint is not None:
                mask_box_ineq = torch.ones(
                    self.ineq_constraint.n_constraints, dtype=torch.bool
                )
                box_lb_ineq = self.ineq_constraint.lb
                box_ub_ineq = self.ineq_constraint.ub
                n_curr += self.ineq_constraint.n_constraints
            else:
                mask_box_ineq = torch.zeros(0, dtype=torch.bool)
                box_lb_ineq = torch.full((1, 0, 1), float("-inf"))
                box_ub_ineq = torch.full((1, 0, 1), float("inf"))
            mask_box_other = torch.zeros(
                n_tot - mask_box_init.numel() - mask_box_ineq.numel(),
                dtype=torch.bool,
            )
            mask_box_lifted = torch.cat(
                [mask_box_init, mask_box_ineq, mask_box_other], dim=0
            )
            box_lb_lifted = torch.cat([box_lb_init, box_lb_ineq], dim=1)
            box_ub_lifted = torch.cat([box_ub_init, box_ub_ineq], dim=1)
            box_lifted = BoxConstraint(
                lb=box_lb_lifted, ub=box_ub_lifted, mask=mask_box_lifted
            )

        for nl in nl_constraints:
            n_u = int(nl.a_mat.shape[1])
            mask_u = torch.zeros(n_tot, dtype=torch.bool)
            mask_t = torch.zeros(n_tot, dtype=torch.bool)
            mask_u[n_curr : n_curr + n_u] = True
            mask_t[n_curr + n_u] = True
            prim_constraints.append(
                SocConstraint(mask_u=mask_u, mask_t=mask_t, a=nl.a, b=nl.b)
            )
            n_curr += n_u + 1

        cartesian_lifted = CartesianConstraint(
            box_constraint=box_lifted, nl_constraints=prim_constraints
        )
        aux_map = lifted_a_b1[:, eq_constraint.n_constraints :, :]

        def lift(y: Tensor) -> Tensor:
            return torch.cat([y, aux_map @ y], dim=1)

        return (eq_lifted, cartesian_lifted, lift, aux_map)
