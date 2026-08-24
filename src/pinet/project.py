"""Implementation of the projection layer."""

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from ._typing import BatchedScalar, ColScaling, RowScaling, ScalarLike
from .constants import Constants
from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
)
from .dataclasses import (
    BoxConstraintSpecification,
    EquilibrationParams,
    ProjectionInstance,
)
from .equilibration import ruiz_equilibration
from .solver import initialize, make_admm_kernels
from .solver.acceleration import (
    AccelCarry,
    AdaptiveCarry,
    AdmmRawStep,
    accelerated_loop,
    adaptive_loop,
    identity_raw_step,
    init_accel_carry,
    init_adaptive_carry,
)

PROJECTION_DEFAULT_SIGMA = Constants.PROJECTION_DEFAULT_SIGMA
PROJECTION_DEFAULT_OMEGA = Constants.PROJECTION_DEFAULT_OMEGA
PROJECTION_DEFAULT_CHECK_EVERY = Constants.PROJECTION_DEFAULT_CHECK_EVERY
PROJECTION_DEFAULT_TOL = Constants.PROJECTION_DEFAULT_TOL
PROJECTION_DEFAULT_MAX_ITER = Constants.PROJECTION_DEFAULT_MAX_ITER
PROJECTION_DEFAULT_CHECK_REDUCTION = Constants.PROJECTION_DEFAULT_CHECK_REDUCTION
PROJECTION_DEFAULT_ANDERSON = Constants.PROJECTION_DEFAULT_ANDERSON
PROJECTION_ANDERSON_MEMORY = Constants.PROJECTION_ANDERSON_MEMORY
PROJECTION_ANDERSON_MIN_HISTORY = Constants.PROJECTION_ANDERSON_MIN_HISTORY
PROJECTION_DEFAULT_ADAPTIVE_PENALTY = Constants.PROJECTION_DEFAULT_ADAPTIVE_PENALTY


class Project:
    """Projection layer implemented via Douglas-Rachford.

    Attributes:
        eq_constraint: Equality constraint.
        ineq_constraint: Affine inequality constraint.
        box_constraint: Box constraint.
        nl_constraints: List of non-linear constraints.
        unroll: Use loop unrolling for backpropagation.
        equilibration_params: Parameters for equilibration.
        use_anderson: Mix Douglas-Rachford iterates with Type-II Anderson
            acceleration.
        use_adaptive_penalty: Residual-balance the ADMM penalty ``sigma``.
        anderson_memory: Number of past iterates mixed by Anderson.
    """

    eq_constraint: EqualityConstraint | None = None
    ineq_constraint: AffineInequalityConstraint | None = None
    box_constraint: BoxConstraint | None = None
    nl_constraints: list[NonLinearConstraint] | None = None
    unroll: bool = False
    equilibration_params: EquilibrationParams | None = None
    use_anderson: bool = PROJECTION_DEFAULT_ANDERSON
    use_adaptive_penalty: bool = PROJECTION_DEFAULT_ADAPTIVE_PENALTY
    anderson_memory: int = PROJECTION_ANDERSON_MEMORY

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None = None,
        ineq_constraint: AffineInequalityConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[NonLinearConstraint] | None = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams | None = None,
        use_anderson: bool = PROJECTION_DEFAULT_ANDERSON,
        use_adaptive_penalty: bool = PROJECTION_DEFAULT_ADAPTIVE_PENALTY,
        anderson_memory: int = PROJECTION_ANDERSON_MEMORY,
    ) -> None:
        """Initialize projection layer.

        Args:
            eq_constraint: Equality constraint.
            ineq_constraint: Affine inequality constraint.
            box_constraint: Box constraint.
            nl_constraints: List of non-linear constraints.
            unroll: Use loop unrolling for backpropagation.
            equilibration_params: Parameters for equilibration.
            use_anderson: Mix Douglas-Rachford iterates with Type-II
                Anderson acceleration.
            use_adaptive_penalty: Residual-balance the ADMM penalty
                ``sigma``.
            anderson_memory: Number of past iterates mixed by Anderson.
        """
        if use_anderson and anderson_memory < PROJECTION_ANDERSON_MIN_HISTORY:
            raise ValueError(
                "anderson_memory must be at least "
                f"{PROJECTION_ANDERSON_MIN_HISTORY} when Anderson is on."
            )
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.nl_constraints = nl_constraints
        self.unroll = unroll
        self.use_anderson = use_anderson
        self.use_adaptive_penalty = use_adaptive_penalty
        self.anderson_memory = anderson_memory
        if equilibration_params is None:
            self.equilibration_params = EquilibrationParams()
        else:
            self.equilibration_params = equilibration_params
        self.setup()

    def setup(self) -> None:
        """Setup the projection layer."""
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
        # The projection layer is meaningful only if at least one constraint is present.
        assert len(constraints) > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim

        is_single_simple_constraint = (
            self.ineq_constraint is None
            and self.nl_constraints is None
            and len(constraints) == 1
        )
        self.is_single_simple_constraint = is_single_simple_constraint

        self.dim_lifted = self.dim
        self.step_iteration = lambda s_prev, y_raw, sigma, omega: s_prev
        self.raw_step = identity_raw_step
        self.step_final = self._project_single
        self.single_constraint = constraints[0]
        self.d_r = jnp.ones((1, self.single_constraint.n_constraints, 1))
        self.d_c = jnp.ones((1, self.single_constraint.dim, 1))
        if not self.is_single_simple_constraint:
            if self.nl_constraints is None:
                # Constraints need to be parsed
                if self.ineq_constraint is not None:
                    self.dim_lifted += self.ineq_constraint.n_constraints
                parser = ConstraintParser(
                    eq_constraint=self.eq_constraint,
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                )
                (parsed_eq, parsed_primitive, self.lift) = parser.parse(method=None)
                # Inequality-constrained parsing must yield a lifted equality.
                assert parsed_eq is not None
                # Inequality-constrained parsing must yield a lifted primitive.
                assert parsed_primitive is not None
                # Setup always stores equilibration parameters before this branch runs.
                assert self.equilibration_params is not None
                # Only equilibrate when we have a single a_mat.
                if not parsed_eq.var_a_mat and parsed_eq.a_mat.shape[0] == 1:
                    scaled_a_mat_flat, d_r_flat, d_c_flat = ruiz_equilibration(
                        parsed_eq.a_mat[0], self.equilibration_params
                    )
                    scaled_a_mat = scaled_a_mat_flat.reshape(
                        1, parsed_eq.a_mat.shape[1], parsed_eq.a_mat.shape[2]
                    )
                    self.d_r = d_r_flat.reshape(1, -1, 1)
                    self.d_c = d_c_flat.reshape(1, -1, 1)
                else:
                    # No equilibration for variable a_mat
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
                    scaled_a_mat = parsed_eq.a_mat
                    self.d_r = jnp.ones((1, n_eq + n_ineq, 1))
                    self.d_c = jnp.ones((1, self.dim_lifted, 1))

                # Build the scaled lifted equality constraint with method="pinv".
                self.lifted_eq_constraint = EqualityConstraint(
                    a_mat=scaled_a_mat,
                    b=parsed_eq.b * self.d_r,
                    method="pinv",
                    var_b=parsed_eq.var_b,
                    var_a_mat=parsed_eq.var_a_mat,
                )

                # Scale the lifted primitive constraint.
                # The polytope path always produces a BoxConstraint (not a cartesian).
                assert isinstance(parsed_primitive, BoxConstraint)
                # BoxConstraint.__init__ guarantees mask/lb/ub are set.
                assert parsed_primitive.mask is not None
                assert parsed_primitive.lb is not None
                assert parsed_primitive.ub is not None
                mask = parsed_primitive.mask
                box_scale = 1 / self.d_c[:, mask, :]

                self.lifted_primitive_constraint = BoxConstraint(
                    BoxConstraintSpecification(
                        lb=parsed_primitive.lb * box_scale,
                        ub=parsed_primitive.ub * box_scale,
                        mask=parsed_primitive.mask,
                    ),
                    scale=box_scale,
                )

                self.raw_step, self.step_iteration, self.step_final = make_admm_kernels(
                    self.lifted_eq_constraint,
                    self.lifted_primitive_constraint,
                    self.dim,
                    self.d_c[:, : self.dim, :],
                )
            else:
                if self.ineq_constraint is not None:
                    self.dim_lifted += self.ineq_constraint.n_constraints
                for nl in self.nl_constraints:
                    self.dim_lifted += nl.a_mat.shape[1]
                    if nl.f is not None:
                        self.dim_lifted += 1

                parser = ConstraintParser(
                    eq_constraint=self.eq_constraint,
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                    nl_constraints=self.nl_constraints,
                )
                (
                    self.lifted_eq_constraint,
                    self.lifted_primitive_constraint,
                    self.lift,
                ) = parser.parse(method="pinv")
                # The non-linear path must produce a lifted equality and a cartesian.
                assert self.lifted_eq_constraint is not None
                assert self.lifted_primitive_constraint is not None
                # Impose no rescaling
                self.d_r = jnp.ones((1, self.lifted_eq_constraint.a_mat.shape[1], 1))
                self.d_c = jnp.ones((1, self.dim_lifted, 1))

                self.raw_step, self.step_iteration, self.step_final = make_admm_kernels(
                    eq_constraint=self.lifted_eq_constraint,
                    box_constraint=self.lifted_primitive_constraint,
                    dim=self.dim,
                    scale=self.d_c[:, : self.dim, :],
                )

        if is_single_simple_constraint:
            # A single simple constraint -- exactly one of an equality or a box
            # constraint -- has a closed-form projection that _project_single
            # applies directly: proj(x) = x - A^+ (A x - b) for the equality
            # case (A is ``a_mat``, A^+ is ``a_mat_pinv``, b is ``b``), or a
            # clamp for the box case.
            # The ADMM initializer zeros out y_raw.x, causing _project_single
            # to project the origin rather than the actual input point.
            # Override initialize so y_raw.x is preserved end-to-end.
            self.initialize = lambda y_raw: y_raw

        project_fn = (
            _project_general
            if (self.unroll or self.is_single_simple_constraint)
            else _project_general_custom
        )

        static_args = (
            ["n_iter"]
            if (self.unroll or self.is_single_simple_constraint)
            else ["n_iter", "n_iter_bwd", "fpi"]
        )

        self._project = jax.jit(
            partial(
                project_fn,
                initialize_fn=self.initialize,
                step_iteration=self.step_iteration,
                step_final=self.step_final,
                dim_lifted=self.dim_lifted,
                raw_step=self.raw_step,
                use_anderson=self.use_anderson,
                use_adaptive_penalty=self.use_adaptive_penalty,
                anderson_memory=self.anderson_memory,
                d_r=self.d_r,
                d_c=self.d_c,
            ),
            static_argnames=static_args,
        )

        # jit correctly the call method
        self.call = self._project

    def initialize(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Returns a zero initial value for the governing sequence.

        Args:
            y_raw: Point to be projected data.

        Returns:
            ProjectionInstance: Initial value for the governing sequence.
        """
        return initialize(
            y_raw=y_raw,
            ineq_constraint=self.ineq_constraint,
            box_constraint=self.box_constraint,
            dim=self.dim,
            dim_lifted=self.dim_lifted,
            d_r=self.d_r,
            nl_constraints=self.nl_constraints,
        )

    def cv(self, y: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            y: Point to be evaluated.

        Returns:
            Constraint violation for each point in the batch.
        """
        if self.is_single_simple_constraint:
            return self.single_constraint.cv(y)

        assert self.lifted_eq_constraint is not None
        assert self.lifted_primitive_constraint is not None
        if y.x.shape[1] != self.dim_lifted:
            y = self.lift(y)
        return jnp.maximum(
            self.lifted_eq_constraint.cv(y),
            self.lifted_primitive_constraint.cv(y),
        )

    def call_and_check(
        self,
        sigma: ScalarLike = PROJECTION_DEFAULT_SIGMA,
        omega: ScalarLike = PROJECTION_DEFAULT_OMEGA,
        check_every: int = PROJECTION_DEFAULT_CHECK_EVERY,
        tol: float = PROJECTION_DEFAULT_TOL,
        max_iter: int = PROJECTION_DEFAULT_MAX_ITER,
        reduction: str | float = PROJECTION_DEFAULT_CHECK_REDUCTION,
        use_anderson: bool = False,
        use_adaptive_penalty: bool = True,
    ) -> Callable[
        [ProjectionInstance], tuple[ProjectionInstance, jax.Array, jax.Array | int]
    ]:
        """Returns a function that projects input and checks constraint violation.

        Args:
            sigma: ADMM parameter.
            omega: ADMM parameter.
            check_every: Frequency of checking constraint violation.
            tol: Tolerance for constraint violation.
            max_iter: Maximum number of iterations for checking.
            reduction: Method to reduce constraint violations among a batch.
                Valid options are "max" (maximum cv less than tol),
                "mean" (mean cv less than tol), or a float in (0, 1)
                (fraction of instances with cv less than tol).
            use_anderson: Mix iterates with Type-II Anderson acceleration.
                Off by default: mixing reduces iterations but the extra
                kernels are rarely worth it on CPU. Pass True to opt in.
            use_adaptive_penalty: Residual-balance the ADMM penalty
                ``sigma``. On by default so a poorly scaled penalty is
                corrected; when ``sigma`` is already well chosen the
                extra work is a cheap residual check every few steps.

        Returns:
            Callable: Takes as input the points to be projected and any
                specifications for the constraints (e.g., the value of b for
                variable b equality constraints). Returns an approximately
                projected point and a flag showing whether the termination
                condition was satisfied.
        """

        @jax.jit
        def check(inp: ProjectionInstance) -> jax.Array:
            if reduction == "max":
                return jnp.max(self.cv(inp)) < tol
            elif reduction == "mean":
                return jnp.mean(self.cv(inp)) < tol
            elif isinstance(reduction, float) and 0 < reduction < 1:
                return jnp.mean(self.cv(inp) < tol) >= reduction
            else:
                raise ValueError(
                    f"Invalid reduction method {reduction}. "
                    "Valid options are: 'max', 'mean', or a float in (0, 1)."
                )

        # Device-side ``while_loop`` avoids host sync on every feasibility
        # check. Acceleration flags only change the inner step; the original
        # solver uses the same fused outer loop with the unaccelerated map.
        hist_memory = self.anderson_memory if use_anderson else 1
        raw_step = self.raw_step
        step_iteration = self.step_iteration
        step_final = self.step_final
        d_c = self.d_c
        initialize_fn = self.initialize
        if use_anderson and (not self.is_single_simple_constraint):

            @jax.jit
            def project_and_check_aa(
                y_raw: ProjectionInstance,
            ) -> tuple[ProjectionInstance, jax.Array, jax.Array]:
                s0 = initialize_fn(y_raw)
                carry0 = init_accel_carry(s0, sigma, hist_memory)

                def cond(
                    state: tuple[ProjectionInstance, AccelCarry, jax.Array, jax.Array],
                ) -> jax.Array:
                    _, _, it, done = state
                    return jnp.logical_and(jnp.logical_not(done), it < max_iter)

                def body(
                    state: tuple[ProjectionInstance, AccelCarry, jax.Array, jax.Array],
                ) -> tuple[ProjectionInstance, AccelCarry, jax.Array, jax.Array]:
                    _, carry_in, it, _ = state
                    carry_out = accelerated_loop(
                        raw_step,
                        carry_in,
                        y_raw,
                        omega,
                        check_every,
                        use_anderson=True,
                        use_adaptive_penalty=use_adaptive_penalty,
                        anderson_memory=hist_memory,
                    )
                    xproj = _finalize_projection(step_final, d_c, carry_out.s, y_raw)
                    done = check(xproj)
                    return xproj, carry_out, it + jnp.int32(check_every), done

                xproj, _, it, done = jax.lax.while_loop(
                    cond,
                    body,
                    (y_raw, carry0, jnp.int32(0), jnp.array(False)),
                )
                return xproj, done, it

            return project_and_check_aa

        if use_adaptive_penalty and (not self.is_single_simple_constraint):

            @jax.jit
            def project_and_check_ad(
                y_raw: ProjectionInstance,
            ) -> tuple[ProjectionInstance, jax.Array, jax.Array]:
                s0 = initialize_fn(y_raw)
                carry0 = init_adaptive_carry(s0, sigma)

                def cond(
                    state: tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array],
                ) -> jax.Array:
                    _, _, it, done = state
                    return jnp.logical_and(jnp.logical_not(done), it < max_iter)

                def body(
                    state: tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array],
                ) -> tuple[ProjectionInstance, AdaptiveCarry, jax.Array, jax.Array]:
                    _, carry_in, it, _ = state
                    carry_out = adaptive_loop(
                        raw_step, carry_in, y_raw, omega, check_every
                    )
                    xproj = _finalize_projection(step_final, d_c, carry_out.s, y_raw)
                    done = check(xproj)
                    return xproj, carry_out, it + jnp.int32(check_every), done

                xproj, _, it, done = jax.lax.while_loop(
                    cond,
                    body,
                    (y_raw, carry0, jnp.int32(0), jnp.array(False)),
                )
                return xproj, done, it

            return project_and_check_ad

        @jax.jit
        def project_and_check_orig(
            y_raw: ProjectionInstance,
        ) -> tuple[ProjectionInstance, jax.Array, jax.Array]:
            s0 = initialize_fn(y_raw)

            def cond(
                state: tuple[
                    ProjectionInstance, ProjectionInstance, jax.Array, jax.Array
                ],
            ) -> jax.Array:
                _, _, it, done = state
                return jnp.logical_and(jnp.logical_not(done), it < max_iter)

            def body(
                state: tuple[
                    ProjectionInstance, ProjectionInstance, jax.Array, jax.Array
                ],
            ) -> tuple[ProjectionInstance, ProjectionInstance, jax.Array, jax.Array]:
                _, s_in, it, _ = state
                s_out, _ = jax.lax.scan(
                    lambda s_prev, _: (
                        step_iteration(s_prev, y_raw, sigma, omega),
                        None,
                    ),
                    s_in,
                    None,
                    length=check_every,
                )
                xproj = _finalize_projection(step_final, d_c, s_out, y_raw)
                done = check(xproj)
                return xproj, s_out, it + jnp.int32(check_every), done

            xproj, _, it, done = jax.lax.while_loop(
                cond,
                body,
                (y_raw, s0, jnp.int32(0), jnp.array(False)),
            )
            return xproj, done, it

        return project_and_check_orig

    def _project_single(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project a batch of points with single constraint.

        Args:
            y_raw: Point to be projected.
                Shape (batch_size, dimension, 1).

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        if y_raw.eq and y_raw.eq.a_mat is not None:
            a_mat_pinv = jnp.linalg.pinv(y_raw.eq.a_mat)
            y_raw = y_raw.update(eq=y_raw.eq.update(a_mat_pinv=a_mat_pinv))

        return self.single_constraint.project(y_raw)


def _finalize_projection(
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    d_c: ColScaling,
    sk: ProjectionInstance,
    y_raw: ProjectionInstance,
) -> ProjectionInstance:
    """Unscale the equality-block projection back to the original variables.

    Args:
        step_final: Equality-block projector.
        d_c: Column scaling of the lifted problem.
        sk: Governing sequence iterate.
        y_raw: Original projection request (used for batch shape and specs).

    Returns:
        Projected point in the original coordinates.
    """
    primal_dim = y_raw.x.shape[1]
    y = step_final(sk).x[:, :primal_dim, :]
    return y_raw.update(x=y * d_c[:, :primal_dim, :])


def _advance_governing_sequence(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
    raw_step: AdmmRawStep,
    y_raw: ProjectionInstance,
    s0: ProjectionInstance | None,
    sigma: ScalarLike,
    omega: ScalarLike,
    n_iter: int,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int,
) -> tuple[ProjectionInstance, ScalarLike]:
    """Run ``n_iter`` Douglas-Rachford steps, optionally accelerated.

    Args:
        initialize_fn: Governing-sequence initializer.
        step_iteration: Unaccelerated ADMM step.
        raw_step: ADMM step that also returns the two blocks.
        y_raw: Point to be projected.
        s0: Optional warm start for the governing sequence.
        sigma: ADMM penalty.
        omega: Relaxation parameter.
        n_iter: Number of iterations to run.
        use_anderson: Enable Type-II Anderson mixing.
        use_adaptive_penalty: Enable residual balancing of ``sigma``.
        anderson_memory: Anderson history length.

    Returns:
        The governing sequence and the penalty used on the last step.
    """
    assert n_iter > 0, "Number of iterations must be positive."
    s0 = initialize_fn(y_raw) if s0 is None else s0
    if use_anderson:
        carry = init_accel_carry(s0, sigma, anderson_memory)
        carry = accelerated_loop(
            raw_step,
            carry,
            y_raw,
            omega,
            n_iter,
            use_anderson=True,
            use_adaptive_penalty=use_adaptive_penalty,
            anderson_memory=anderson_memory,
        )
        return carry.s, carry.sigma
    if use_adaptive_penalty:
        carry_ad = init_adaptive_carry(s0, sigma)
        carry_ad = adaptive_loop(raw_step, carry_ad, y_raw, omega, n_iter)
        return carry_ad.s, carry_ad.sigma

    sk, _ = jax.lax.scan(
        lambda s_prev, _: (
            step_iteration(s_prev, y_raw, sigma, omega),
            None,
        ),
        s0,
        None,
        length=n_iter,
    )
    return sk, sigma


# Project general
def _project_general(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    raw_step: AdmmRawStep,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int,
    d_r: RowScaling,
    d_c: ColScaling,
    y_raw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: ScalarLike = PROJECTION_DEFAULT_SIGMA,
    omega: ScalarLike = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 100,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    """Project a batch of points using Douglas-Rachford.

    Args:
        initialize_fn: Function to initialize the governing sequence.
        step_iteration: Function for the iteration step.
        step_final: Function for the final step.
        dim_lifted: Dimension of the lifted space.
        raw_step: ADMM step that also returns the two blocks.
        use_anderson: Mix iterates with Type-II Anderson acceleration.
        use_adaptive_penalty: Residual-balance the ADMM penalty.
        anderson_memory: Anderson history length.
        d_r: Scaling factor for the rows.
        d_c: Scaling factor for the columns.
        y_raw: Point to be projected.
        s0: Initial value for the governing sequence.
        sigma: ADMM parameter.
        omega: ADMM parameter.
        n_iter: Number of iterations to run.

    Returns:
        A pair ``(projected_point, governing_sequence_value)``.
    """
    del dim_lifted, d_r
    assert n_iter > 0, "Number of iterations must be positive."
    sk, _ = _advance_governing_sequence(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        raw_step=raw_step,
        y_raw=y_raw,
        s0=s0,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
        use_anderson=use_anderson,
        use_adaptive_penalty=use_adaptive_penalty,
        anderson_memory=anderson_memory,
    )
    return _finalize_projection(step_final, d_c, sk, y_raw), sk


@partial(
    jax.custom_vjp,
    nondiff_argnames=[
        "initialize_fn",
        "step_iteration",
        "step_final",
        "dim_lifted",
        "raw_step",
        "use_anderson",
        "use_adaptive_penalty",
        "anderson_memory",
        "n_iter",
        "n_iter_bwd",
        "fpi",
    ],
)
def _project_general_custom(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    raw_step: AdmmRawStep,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int,
    d_r: RowScaling,
    d_c: ColScaling,
    y_raw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: ScalarLike = PROJECTION_DEFAULT_SIGMA,
    omega: ScalarLike = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    return _project_general(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        step_final=step_final,
        dim_lifted=dim_lifted,
        raw_step=raw_step,
        use_anderson=use_anderson,
        use_adaptive_penalty=use_adaptive_penalty,
        anderson_memory=anderson_memory,
        d_r=d_r,
        d_c=d_c,
        s0=s0,
        y_raw=y_raw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )


def _project_general_fwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    raw_step: AdmmRawStep,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int,
    d_r: RowScaling,
    d_c: ColScaling,
    y_raw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: ScalarLike = PROJECTION_DEFAULT_SIGMA,
    omega: ScalarLike = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[
    tuple[ProjectionInstance, ProjectionInstance],
    tuple[
        ProjectionInstance,
        ProjectionInstance,
        RowScaling,
        ColScaling,
        ScalarLike,
        ScalarLike,
    ],
]:
    del dim_lifted, n_iter_bwd, fpi
    sk, sigma_final = _advance_governing_sequence(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        raw_step=raw_step,
        y_raw=y_raw,
        s0=s0,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
        use_anderson=use_anderson,
        use_adaptive_penalty=use_adaptive_penalty,
        anderson_memory=anderson_memory,
    )
    y = _finalize_projection(step_final, d_c, sk, y_raw)
    return (y, sk), (sk, y_raw, d_r, d_c, sigma_final, omega)


def _project_general_bwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    raw_step: AdmmRawStep,
    use_anderson: bool,
    use_adaptive_penalty: bool,
    anderson_memory: int,
    n_iter: int,
    n_iter_bwd: int,
    fpi: bool,
    residuals: tuple[
        ProjectionInstance,
        ProjectionInstance,
        RowScaling,
        ColScaling,
        ScalarLike,
        ScalarLike,
    ],
    cotangent: tuple[ProjectionInstance, ProjectionInstance],
) -> tuple[None, None, ProjectionInstance, None, None, None]:
    """Backward pass for custom vjp.

    This function computes the vjp for the projection using the
    implicit function theorem.
    Note that, the arguments are:
    (i) any arguments for the
    forward that are not arrays;
    (ii) residuals: tuple with auxiliary data from the forward pass;
    (iii) cotangent: incoming cotangents.
    The function returns a tuple where each element corresponds
    to an array from the input.

    Args:
        initialize_fn: Function to initialize the governing sequence.
        step_iteration: Function for the iteration step.
        step_final: Function for the final step.
        dim_lifted: Dimension of the lifted space.
        raw_step: ADMM step that also returns the two blocks.
        use_anderson: Unused; the VJP differentiates the unaccelerated map.
        use_adaptive_penalty: Unused; the VJP uses the frozen final penalty.
        anderson_memory: Unused Anderson history length.
        n_iter: Number of iterations to run.
        n_iter_bwd: Number of iterations for backward pass.
        fpi: Whether to use fixed-point iteration.
        residuals: Auxiliary data from the forward pass.
        cotangent: Incoming cotangents.

    Returns:
        tuple: The computed cotangent for the projection.
    """
    del initialize_fn, raw_step, use_anderson, use_adaptive_penalty
    del anderson_memory, n_iter
    s_k, y_raw, _, d_c, sigma, omega = residuals
    cotangent_zk1, _ = cotangent

    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(xx, y_raw, sigma, omega),
        s_k,
    )
    _, iteration_vjp2 = jax.vjp(lambda xx: step_iteration(s_k, xx, sigma, omega), y_raw)
    _, equality_vjp = jax.vjp(step_final, s_k)

    # Rescale the gradient
    cotangent_zk1 = cotangent_zk1.x * d_c[:, : y_raw.x.shape[1], :]

    # Compute VJP of cotangent with projection before auxiliary
    cotangent_eq_6 = equality_vjp(
        s_k.update(
            x=jnp.concatenate(
                [
                    cotangent_zk1,
                    jnp.zeros(
                        (cotangent_zk1.shape[0], dim_lifted - cotangent_zk1.shape[1], 1)
                    ),
                ],
                axis=1,
            )
        )
    )[0].x
    # Run iteration
    if fpi:

        def body_fn(x, _):
            vjp = iteration_vjp(x)[0].x
            return s_k.update(x=(vjp + cotangent_eq_6)), None

        cotangent_eq_7, _ = jax.lax.scan(
            body_fn,
            s_k.update(x=jnp.zeros((cotangent_zk1.shape[0], dim_lifted, 1))),
            None,
            length=n_iter_bwd,
        )
    else:
        cotangent_eq_7 = jax.scipy.sparse.linalg.bicgstab(
            lambda x: x - iteration_vjp(s_k.update(x=x))[0].x,
            cotangent_eq_6,
            maxiter=n_iter_bwd,
        )[0]
        cotangent_eq_7 = s_k.update(x=cotangent_eq_7)

    thevjp = iteration_vjp2(cotangent_eq_7)[0]

    return (None, None, thevjp, None, None, None)


_project_general_custom.defvjp(_project_general_fwd, _project_general_bwd)
