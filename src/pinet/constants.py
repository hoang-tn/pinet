"""Constants used throughout the pinet package."""


class Constants:
    """Centralized constants used throughout the pinet package.

    Attributes:
        SOC_CONSTRAINT_EPSILON: Epsilon for SOC norm divisions.
        EQUILIBRATION_DEFAULT_TOL: Default equilibration termination tolerance.
        EQUILIBRATION_DEFAULT_MAX_ITER: Default max equilibration iterations.
        EQUILIBRATION_DEFAULT_ORD: Default norm order for equilibration.
        EQUILIBRATION_DEFAULT_COL_SCALING: Whether to apply column scaling.
        EQUILIBRATION_DEFAULT_UPDATE_MODE: Equilibration update mode.
        EQUILIBRATION_DEFAULT_SAFEGUARD: Whether to safeguard equilibration.
        PROJECTION_DEFAULT_SIGMA: Default ADMM stepsize parameter.
        PROJECTION_DEFAULT_OMEGA: Default ADMM relaxation parameter.
        PROJECTION_DEFAULT_CHECK_EVERY: Default constraint-check period.
        PROJECTION_DEFAULT_TOL: Default constraint-violation tolerance.
        PROJECTION_DEFAULT_MAX_ITER: Default max projection iterations.
        PROJECTION_DEFAULT_CHECK_REDUCTION: Default batch reduction method.
        PROJECTION_DEFAULT_ANDERSON: Default Anderson acceleration flag.
        PROJECTION_ANDERSON_MEMORY: Anderson history length.
        PROJECTION_ANDERSON_MIN_HISTORY: Minimum history needed to mix.
        PROJECTION_ANDERSON_REGULARIZATION: Anderson least-squares regularization.
        PROJECTION_ANDERSON_SAFEGUARD: Anderson residual-ratio safeguard.
        PROJECTION_ANDERSON_EVERY: Apply Anderson every this many steps.
        PROJECTION_ANDERSON_STOP_RESIDUAL: Disable mixing below this residual.
        PROJECTION_DEFAULT_ADAPTIVE_PENALTY: Default residual-balancing flag.
        PROJECTION_ADAPTIVE_PENALTY_MU: Residual-ratio threshold for penalty updates.
        PROJECTION_ADAPTIVE_PENALTY_TAU: Multiplicative penalty step.
        PROJECTION_ADAPTIVE_PENALTY_EVERY: Penalty update period.
        PROJECTION_ADAPTIVE_PENALTY_SIGMA_MIN: Lower bound on adapted sigma.
        PROJECTION_ADAPTIVE_PENALTY_SIGMA_MAX: Upper bound on adapted sigma.
    """

    # Numerical Stability
    # ==================
    # Epsilon for protecting against division by zero in SOC norm calculations
    SOC_CONSTRAINT_EPSILON: float = 1e-12

    # Equilibration Algorithm Defaults
    # =================================
    # Default tolerance for equilibration termination criterion
    EQUILIBRATION_DEFAULT_TOL: float = 1e-3

    # Default maximum number of equilibration iterations
    EQUILIBRATION_DEFAULT_MAX_ITER: int = 0

    # Default norm order for equilibration (1, 2, or inf)
    EQUILIBRATION_DEFAULT_ORD: float = 2.0

    # Enable column scaling in equilibration
    EQUILIBRATION_DEFAULT_COL_SCALING: bool = False

    # Update mode for equilibration: "Gauss" (sequential) or "Jacobi" (simultaneous)
    EQUILIBRATION_DEFAULT_UPDATE_MODE: str = "Gauss"

    # Enable safeguard to ensure condition number doesn't increase
    EQUILIBRATION_DEFAULT_SAFEGUARD: bool = False

    # Projection/ADMM Algorithm Parameters
    # =====================================
    # Default ADMM stepsize parameter
    PROJECTION_DEFAULT_SIGMA: float = 1.0

    # Default ADMM relaxation parameter (typically between 1 and 2)
    PROJECTION_DEFAULT_OMEGA: float = 1.7

    # Default frequency of constraint violation checks during projection
    PROJECTION_DEFAULT_CHECK_EVERY: int = 10

    # Default tolerance for constraint violation during projection
    PROJECTION_DEFAULT_TOL: float = 1e-3

    # Default maximum number of projection iterations
    PROJECTION_DEFAULT_MAX_ITER: int = 100

    # Default reduction method for batch constraint checks:
    # "max", "mean", or a float in [0, 1]
    PROJECTION_DEFAULT_CHECK_REDUCTION: str = "max"

    # Enable Type-II Anderson acceleration of the Douglas-Rachford
    # governing sequence. Off by default so ``call(n_iter=...)`` stays
    # bit-identical to the original solver. ``call_and_check`` also
    # leaves Anderson off: mixing reduces iterations but is rarely a
    # wall-clock win on CPU. Pass ``use_anderson=True`` to opt in.
    PROJECTION_DEFAULT_ANDERSON: bool = False

    # Number of past iterates mixed by Anderson acceleration.
    # Memory 2 is Type-II AA(1): a closed-form secant step, cheap enough
    # that mixing every iteration does not dominate the two projections.
    PROJECTION_ANDERSON_MEMORY: int = 2

    # Minimum history length required to form an Anderson mix.
    PROJECTION_ANDERSON_MIN_HISTORY: int = 2

    # Tikhonov regularization for the Anderson least-squares Gram matrix.
    PROJECTION_ANDERSON_REGULARIZATION: float = 1e-4

    # Reject the Anderson candidate when its residual exceeds this
    # multiple of the unaccelerated residual.
    PROJECTION_ANDERSON_SAFEGUARD: float = 2.0

    # Apply Anderson mixing every this many Douglas-Rachford steps.
    # Every other step is a good tradeoff: AA(1) still cuts iterations
    # while the unaccelerated step stays a pure projection pair.
    PROJECTION_ANDERSON_EVERY: int = 2

    # Stop mixing once the mean fixed-point residual is below this
    # threshold, so late-stage Anderson cannot overshoot the tolerance.
    PROJECTION_ANDERSON_STOP_RESIDUAL: float = 1e-3

    # Enable residual-balancing updates of the ADMM penalty ``sigma``.
    # Off by default on ``Project`` / ``call()`` so training stays
    # bit-identical; ``call_and_check`` turns this on unless the caller
    # opts out.
    PROJECTION_DEFAULT_ADAPTIVE_PENALTY: bool = False

    # Residual-ratio threshold for penalty updates (Boyd et al.).
    PROJECTION_ADAPTIVE_PENALTY_MU: float = 4.0

    # Multiplicative penalty step when residuals are unbalanced.
    PROJECTION_ADAPTIVE_PENALTY_TAU: float = 2.0

    # Update ``sigma`` every this many iterations.
    PROJECTION_ADAPTIVE_PENALTY_EVERY: int = 5

    # Bounds on the adapted penalty.
    PROJECTION_ADAPTIVE_PENALTY_SIGMA_MIN: float = 1e-4
    PROJECTION_ADAPTIVE_PENALTY_SIGMA_MAX: float = 1e2
