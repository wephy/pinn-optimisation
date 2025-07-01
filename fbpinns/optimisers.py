import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from typing import NamedTuple, Callable, Any, Optional

# ----------------------------------------------------------------------------
# 1. Define the Optimizer State
# ----------------------------------------------------------------------------
# The state now includes the current damping factor `mu`, which will be
# adapted at each step based on the success of the previous step.
class AdaptiveNewtonState(NamedTuple):
    """State for the adaptive Newton optimizer."""
    count: jnp.ndarray
    mu: jnp.ndarray  # The adaptive damping factor (trust-region radius)

# ----------------------------------------------------------------------------
# 2. Optimizer "Factory" Function
# ----------------------------------------------------------------------------
def adaptive_newton_method(
    learning_rate: float = 1.0,
    initial_damping: float = 1e-4,
    min_damping: float = 1e-6,
    damping_increase_factor: float = 2.0,
    damping_decrease_factor: float = 3.0,
    min_gain_ratio: float = 1e-3,
    max_update_norm: Optional[float] = 1000.0
) -> optax.GradientTransformation:
    """
    A robust trust-region style Newton method with adaptive damping.

    This optimizer calculates a proposed Newton step and only accepts it if it
    actually improves the loss function. If a step is rejected, the damping
    is increased, making the next step smaller and more cautious.

    Args:
        learning_rate: Step-size scaling factor for the Newton direction.
        initial_damping: The starting value for the adaptive damping term.
        min_damping: The minimum value for the damping term.
        damping_increase_factor: Factor to increase damping on bad steps.
        damping_decrease_factor: Factor to decrease damping on good steps.
        min_gain_ratio: The minimum ratio of actual_reduction/predicted_reduction
                        for a step to be considered successful.
        max_update_norm: The maximum allowed L2 norm for the update.

    Returns:
        An optax.GradientTransformation object.
    """

    def init_fn(params: optax.Params) -> AdaptiveNewtonState:
        """Initialises the optimiser state."""
        return AdaptiveNewtonState(
            count=jnp.zeros([], jnp.int32),
            mu=jnp.asarray(initial_damping, dtype=jnp.float32)
        )

    def update_fn(
        grads: optax.Updates,
        state: AdaptiveNewtonState,
        params: optax.Params,
        *,
        value: float,
        value_fn: Callable[[optax.Params], Any],
        **extra_kwargs
    ) -> tuple[optax.Updates, AdaptiveNewtonState]:
        """Performs the Newton update step with an accept/reject mechanism."""
        if value_fn is None or value is None:
            raise ValueError("This optimizer requires `value_fn` and the current `value` (loss).")

        # Flatten parameters and gradients into single vectors
        params_flat, unravel_fn = ravel_pytree(params)
        grads_flat, _ = ravel_pytree(grads)

        def flat_value_fn(p_flat):
            return value_fn(unravel_fn(p_flat))

        # --- Calculate Proposed Step ---
        hessian_matrix = jax.hessian(flat_value_fn)(params_flat)
        num_params = params_flat.shape[0]
        
        # Damp the hessian: H' = H + mu * I
        hessian_damped = hessian_matrix + state.mu * jnp.eye(num_params)
        
        # Solve for the proposed update direction `p`
        # Using Cholesky solve is more stable for positive-definite systems
        try:
            L = jax.scipy.linalg.cholesky(hessian_damped, lower=True)
            p_flat = -jax.scipy.linalg.cho_solve((L, True), grads_flat)
        except jnp.linalg.LinAlgError:
            # Fallback to standard solve if Cholesky fails (not positive-definite)
            p_flat = -jnp.linalg.solve(hessian_damped, grads_flat)

        # --- Evaluate the Quality of the Proposed Step ---
        # Predicted reduction in loss from the quadratic model: m(0) - m(p)
        predicted_reduction = - (jnp.dot(grads_flat, p_flat) + 0.5 * jnp.dot(p_flat, hessian_matrix @ p_flat))
        
        # Actual reduction in loss: f(x) - f(x+p)
        proposed_params = unravel_fn(params_flat + p_flat)
        loss_after_step = value_fn(proposed_params)
        actual_reduction = value - loss_after_step

        # Gain ratio: rho = actual_reduction / predicted_reduction
        gain_ratio = actual_reduction / (predicted_reduction + 1e-8)

        # --- Accept or Reject the Step ---
        is_step_successful = jnp.logical_and(actual_reduction > 0, gain_ratio > min_gain_ratio)

        # If the step is successful, use the proposed update `p`. Otherwise, use zero.
        final_updates_flat = jnp.where(is_step_successful, p_flat, jnp.zeros_like(p_flat))
        
        # Scale by learning rate and clip norm
        final_updates_flat = learning_rate * final_updates_flat
        if max_update_norm is not None:
            update_norm = jnp.linalg.norm(final_updates_flat)
            scale_factor = jnp.minimum(1.0, max_update_norm / (update_norm + 1e-8))
            final_updates_flat = final_updates_flat * scale_factor
        
        final_updates = unravel_fn(jnp.nan_to_num(final_updates_flat))

        # --- Adapt Damping for Next Iteration ---
        # If successful, decrease damping (be more aggressive).
        # If unsuccessful, increase damping (be more cautious).
        new_mu = jnp.where(
            is_step_successful,
            jnp.maximum(min_damping, state.mu / damping_decrease_factor),
            state.mu * damping_increase_factor
        )

        return final_updates, AdaptiveNewtonState(count=state.count + 1, mu=new_mu)

    return optax.GradientTransformation(init_fn, update_fn)
