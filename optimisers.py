import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from functools import partial

from typing import Callable, Any, Dict, List

from jax.flatten_util import ravel_pytree

@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, static_parts, opt_state, optimiser, problem, collocation_points):
    loss, updates, new_opt_state, metrics = optimiser.update(
        params, static_parts, opt_state, problem.physics_residual, collocation_points
    )
    new_params = eqx.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, metrics

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def resample_rad(
    model, candidate_points, n_points_to_sample, problem, k, c, key
):
    # 1. Calculate PDE residuals for all candidate points
    vmapped_residual_fn = jax.vmap(problem.physics_residual, in_axes=(None, 0))
    residuals = vmapped_residual_fn(model, candidate_points)
        
    # 2. Compute sampling probability based on the RAD formula
    epsilon_k = jnp.abs(residuals)**k
    weights = (epsilon_k / jnp.mean(epsilon_k)) + c
    probs = weights / jnp.sum(weights)

    # 3. Sample new points based on the calculated probabilities
    indices = jax.random.choice(
        key, a=candidate_points.shape[0], shape=(n_points_to_sample,), p=probs, replace=False
    )
    
    return candidate_points[indices]


# class AggressiveLM:
#     def __init__(
#         self,
#         init_lambda=1e3,
#         increase_factor: float = 1.05,
#         max_increase: float = 2.0,
#         optimism_factor: float = 0.9,
#         decrease_factor: float = 0.75,
#     ):
#         self.init_lambda = init_lambda
#         self.increase_factor = increase_factor
#         self.max_increase = max_increase
#         self.optimism_factor = optimism_factor
#         self.decrease_factor = decrease_factor

#     def init(self, params: eqx.Module):
#         return {
#             'lambda': jnp.asarray(self.init_lambda),
#             'last_was_optimistic': False
#         }

#     def update_optimism_factor(self, optimism_factor: float):
#         self.optimism_factor = optimism_factor

#     def update(
#         self,
#         params: eqx.Module,
#         static,
#         state: optax.OptState,
#         residual_fn: Callable[[eqx.Module], jax.Array],
#         collocation_points
#     ):
#         current_lambda = state['lambda']
#         last_was_optimistic = state['last_was_optimistic']
        
#         params_flat, unflatten = ravel_pytree(params)
#         num_points = collocation_points.shape[0]
#         num_params = params_flat.shape[0]

#         micro_batch_size = 8
#         macro_batch_size = 512

#         batched_points_for_scan = collocation_points.reshape(
#             (num_points // macro_batch_size, macro_batch_size, -1)
#         )

#         def flat_residual_fn_micro_batch(p_flat_vector, points_micro_batch):
#             params_pytree = unflatten(p_flat_vector)
#             rebuilt_model = eqx.combine(params_pytree, static)
#             residuals_micro_batch = jax.vmap(partial(residual_fn, rebuilt_model))(points_micro_batch)
#             return residuals_micro_batch, residuals_micro_batch

#         compute_J_and_R_micro_batch = jax.jacrev(
#             flat_residual_fn_micro_batch, argnums=0, has_aux=True
#         )

#         def compute_J_and_R_macro_batch(p_flat_vector, points_macro_batch):
#             # Reshape macro-batch into (num_micro_batches, micro_batch_size, dims)
#             num_micro_batches = macro_batch_size // micro_batch_size
#             points_for_vmap = points_macro_batch.reshape(
#                 (num_micro_batches, micro_batch_size, -1)
#             )

#             J_vmapped, R_vmapped = jax.vmap(
#                 compute_J_and_R_micro_batch, in_axes=(None, 0)
#             )(p_flat_vector, points_for_vmap)

#             J_macro = J_vmapped.reshape(macro_batch_size, num_params)
#             R_macro = R_vmapped.reshape(macro_batch_size)
#             return J_macro, R_macro

#         def scan_body(carry, macro_batch_of_points):
#             J_macro, R_macro = compute_J_and_R_macro_batch(params_flat, macro_batch_of_points)
#             return carry, (J_macro, R_macro)

#         _, (all_J_batches, all_residuals_batches) = jax.lax.scan(
#             scan_body, None, batched_points_for_scan
#         )

#         residuals = all_residuals_batches.reshape((num_points,))
#         J = all_J_batches.reshape((num_points, num_params))
        
#         current_loss = jnp.mean(residuals**2)

#         U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
#         g = U.T @ residuals
        
#         def cond_fun(carry):
#             _, accepted, trial_lambda, _ = carry
#             return (trial_lambda < self.max_increase * current_lambda) & (jnp.logical_not(accepted))

#         def body_fun(carry):
#             count, _, trial_lambda, _ = carry
            
#             delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
#             trial_params = unflatten(params_flat + delta_flat)
#             rebuilt_trial_model = eqx.combine(trial_params, static)
#             trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
#             trial_loss = jnp.mean(trial_residuals**2)

#             accepted = trial_loss <= current_loss

#             trial_lambda_new = jax.lax.cond(
#                 accepted,
#                 lambda: trial_lambda,
#                 lambda: trial_lambda * self.increase_factor,
#             )
            
#             return count + 1, accepted, trial_lambda_new, trial_loss

#         init_carry = (0, False, current_lambda, current_loss)
#         count, accepted, trial_lambda, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
#         final_lambda = jax.lax.cond(accepted, lambda: trial_lambda * self.optimism_factor, lambda: trial_lambda)
        
#         delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g

#         updates = unflatten(delta_flat)

#         new_state = {
#             'lambda': jax.lax.cond(jnp.logical_not(last_was_optimistic) & (accepted), lambda: final_lambda * self.decrease_factor, lambda: final_lambda),
#             'last_was_optimistic': jnp.logical_not(last_was_optimistic) & (accepted)
#         }
        
#         metrics = {
#             'loss': current_loss,
#             'singular_values': S,
#             'search_direction_norm': jnp.linalg.norm(final_lambda),
#             'lambda': final_lambda,
#             'trial_count': count
#         }

#         return current_loss, updates, new_state, metrics


def compute_full_J_and_R(
    params_flat: jax.Array,
    static,
    unflatten_fn: Callable,
    residual_fn: Callable,
    collocation_points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Helper function to compute the full Jacobian and residuals with batching."""
    num_points = collocation_points.shape[0]
    num_params = params_flat.shape[0]

    # Note: For efficiency, micro/macro batch sizes could be class attributes
    micro_batch_size = 8
    macro_batch_size = 512

    batched_points_for_scan = collocation_points.reshape(
        (num_points // macro_batch_size, macro_batch_size, -1)
    )

    def flat_residual_fn_micro_batch(p_flat_vector, points_micro_batch):
        params_pytree = unflatten_fn(p_flat_vector)
        rebuilt_model = eqx.combine(params_pytree, static)
        # Use vmap over the points in the micro-batch
        residuals_micro_batch = jax.vmap(
            partial(residual_fn, rebuilt_model)
        )(points_micro_batch)
        return residuals_micro_batch, residuals_micro_batch

    compute_J_and_R_micro_batch = jax.jacrev(
        flat_residual_fn_micro_batch, argnums=0, has_aux=True
    )

    def compute_J_and_R_macro_batch(p_flat_vector, points_macro_batch):
        num_micro_batches = macro_batch_size // micro_batch_size
        points_for_vmap = points_macro_batch.reshape(
            (num_micro_batches, micro_batch_size, -1)
        )
        # vmap the jacobian calculation over the micro-batches
        J_vmapped, R_vmapped = jax.vmap(
            compute_J_and_R_micro_batch, in_axes=(None, 0)
        )(p_flat_vector, points_for_vmap)
        # Reshape to form the macro-batch Jacobian and Residuals
        J_macro = J_vmapped.reshape(macro_batch_size, num_params)
        R_macro = R_vmapped.reshape(macro_batch_size)
        return J_macro, R_macro

    def scan_body(carry, macro_batch_of_points):
        J_macro, R_macro = compute_J_and_R_macro_batch(params_flat, macro_batch_of_points)
        # No carry needed, just collecting results
        return carry, (J_macro, R_macro)

    _, (all_J_batches, all_residuals_batches) = jax.lax.scan(
        scan_body, None, batched_points_for_scan
    )

    residuals = all_residuals_batches.reshape((num_points,))
    J = all_J_batches.reshape((num_points, num_params))
    return J, residuals


class AggressiveLM:
    def __init__(
        self,
        init_lambda=1e3,
        increase_factor: float = 1.05,
        max_increase: float = 2.0,
        optimism_factor: float = 0.5,
        n_times: int = 5,
        decrease_factor: float = 0.75,
    ):
        self.init_lambda = init_lambda
        self.increase_factor = increase_factor
        self.max_increase = max_increase
        self.optimism_factor = optimism_factor
        self.n_times = n_times
        self.decrease_factor = decrease_factor

    def init(self, params: eqx.Module):
        return {
            'lambda': jnp.asarray(self.init_lambda),
        }

    def update_optimism_factor(self, optimism_factor: float):
        self.optimism_factor = optimism_factor

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        current_lambda = state['lambda']
        
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
    
        
        # num_points = collocation_points.shape[0]
        # num_params = params_flat.shape[0]

        # micro_batch_size = 8
        # macro_batch_size = 512

        # batched_points_for_scan = collocation_points.reshape(
        #     (num_points // macro_batch_size, macro_batch_size, -1)
        # )

        # def flat_residual_fn_micro_batch(p_flat_vector, points_micro_batch):
        #     params_pytree = unflatten(p_flat_vector)
        #     rebuilt_model = eqx.combine(params_pytree, static)
        #     residuals_micro_batch = jax.vmap(partial(residual_fn, rebuilt_model))(points_micro_batch)
        #     return residuals_micro_batch, residuals_micro_batch

        # compute_J_and_R_micro_batch = jax.jacrev(
        #     flat_residual_fn_micro_batch, argnums=0, has_aux=True
        # )

        # def compute_J_and_R_macro_batch(p_flat_vector, points_macro_batch):
        #     # Reshape macro-batch into (num_micro_batches, micro_batch_size, dims)
        #     num_micro_batches = macro_batch_size // micro_batch_size
        #     points_for_vmap = points_macro_batch.reshape(
        #         (num_micro_batches, micro_batch_size, -1)
        #     )

        #     J_vmapped, R_vmapped = jax.vmap(
        #         compute_J_and_R_micro_batch, in_axes=(None, 0)
        #     )(p_flat_vector, points_for_vmap)

        #     J_macro = J_vmapped.reshape(macro_batch_size, num_params)
        #     R_macro = R_vmapped.reshape(macro_batch_size)
        #     return J_macro, R_macro

        # def scan_body(carry, macro_batch_of_points):
        #     J_macro, R_macro = compute_J_and_R_macro_batch(params_flat, macro_batch_of_points)
        #     return carry, (J_macro, R_macro)

        # _, (all_J_batches, all_residuals_batches) = jax.lax.scan(
        #     scan_body, None, batched_points_for_scan
        # )

        # residuals = all_residuals_batches.reshape((num_points,))
        # J = all_J_batches.reshape((num_points, num_params))
        
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        def cond_fun(carry):
            _, accepted, trial_lambda, _ = carry
            return (trial_lambda < self.max_increase * current_lambda) & (jnp.logical_not(accepted))

        def body_fun(carry):
            count, _, trial_lambda, _ = carry
            
            delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
            trial_params = unflatten(params_flat + delta_flat)
            rebuilt_trial_model = eqx.combine(trial_params, static)
            trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
            trial_loss = jnp.mean(trial_residuals**2)

            accepted = trial_loss <= current_loss

            trial_lambda_new = jax.lax.cond(
                accepted,
                lambda: trial_lambda,
                lambda: trial_lambda * self.increase_factor,
            )
            
            return count + 1, accepted, trial_lambda_new, trial_loss

        init_carry = (0, False, current_lambda, current_loss)
        count, accepted, trial_lambda, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
        final_lambda = jax.lax.cond(accepted, lambda: trial_lambda * self.optimism_factor, lambda: trial_lambda)
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g

        updates = unflatten(delta_flat)

        new_state = {
            'lambda': jax.lax.cond(accepted, lambda: final_lambda * self.decrease_factor, lambda: final_lambda),
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
            'trial_count': count
        }

        return current_loss, updates, new_state, metrics


class PlateauLM:
    def __init__(
        self,
        init_lambda=1e3,
        increase_factor: float = 1.05,
        decrease_factor: float = 0.75,
        max_increase: float = 2.0,
        plateau_factor: float= 1.05,
    ):
        self.init_lambda = init_lambda
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_increase = max_increase
        self.plateau_factor = plateau_factor

    def init(self, params: eqx.Module):
        return {
            'lambda': jnp.asarray(self.init_lambda),
        }

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        current_lambda = state['lambda']
        
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
        
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        def cond_fun(carry):
            _, accepted, trial_lambda, _, _ = carry
            return (trial_lambda < self.max_increase * current_lambda) & (jnp.logical_not(accepted))

        def body_fun(carry):
            count, _, trial_lambda, previous_loss, previous_plateau = carry
            
            delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
            trial_params = unflatten(params_flat + delta_flat)
            rebuilt_trial_model = eqx.combine(trial_params, static)
            trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
            trial_loss = jnp.mean(trial_residuals**2)

            current_plateau = (self.plateau_factor * trial_loss > previous_loss)

            accepted = (trial_loss <= current_loss) & (previous_plateau) & (current_plateau)

            trial_lambda_new = jax.lax.cond(
                accepted,
                lambda: trial_lambda,
                lambda: trial_lambda * self.increase_factor
            )
            
            return count + 1, accepted, trial_lambda_new, trial_loss, current_plateau

        init_carry = (0, False, current_lambda, current_loss, False)
        count, accepted, trial_lambda, final_loss, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
        final_lambda = jax.lax.cond(accepted, lambda: trial_lambda, lambda: trial_lambda)
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g

        updates = unflatten(delta_flat)

        new_state = {
            'lambda': jax.lax.cond(accepted, lambda: trial_lambda * self.decrease_factor, lambda: trial_lambda,
            ),
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
            'trial_count': count
        }

        return current_loss, updates, new_state, metrics

# class PlateauLM:
#     def __init__(
#         self,
#         init_lambda=1e3,
#         increase_factor: float = 1.05,
#         decrease_factor: float = 0.75,
#         max_increase: float = 2.0,
#         plateau_factor: float= 1.05,
#     ):
#         self.init_lambda = init_lambda
#         self.increase_factor = increase_factor
#         self.decrease_factor = decrease_factor
#         self.max_increase = max_increase
#         self.plateau_factor = plateau_factor

#     def init(self, params: eqx.Module):
#         return {
#             'lambda': jnp.asarray(self.init_lambda),
#         }

#     def update(
#         self,
#         params: eqx.Module,
#         static,
#         state: optax.OptState,
#         residual_fn: Callable[[eqx.Module], jax.Array],
#         collocation_points
#     ):
#         current_lambda = state['lambda']
        
#         params_flat, unflatten = ravel_pytree(params)
        
#         J, residuals = compute_full_J_and_R(
#             params_flat, static, unflatten, residual_fn, collocation_points
#         )
        
#         current_loss = jnp.mean(residuals**2)

#         U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
#         g = U.T @ residuals
        
#         def cond_fun(carry):
#             _, accepted, trial_lambda, _, _ = carry
#             return (trial_lambda < self.max_increase * current_lambda) & (jnp.logical_not(accepted))

#         def body_fun(carry):
#             count, _, trial_lambda, previous_loss, previous_plateau = carry
            
#             delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
#             trial_params = unflatten(params_flat + delta_flat)
#             rebuilt_trial_model = eqx.combine(trial_params, static)
#             trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
#             trial_loss = jnp.mean(trial_residuals**2)

#             current_plateau = (self.plateau_factor * trial_loss > previous_loss)

#             accepted = (trial_loss <= current_loss) & (previous_plateau) & (current_plateau)

#             trial_lambda_new = jax.lax.cond(
#                 accepted,
#                 lambda: trial_lambda,
#                 lambda: trial_lambda * self.increase_factor
#             )
            
#             return count + 1, accepted, trial_lambda_new, trial_loss, current_plateau

#         init_carry = (0, False, current_lambda, current_loss, False)
#         count, accepted, trial_lambda, final_loss, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
#         final_lambda = jax.lax.cond(accepted, lambda: trial_lambda, lambda: trial_lambda)
        
#         delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g

#         updates = unflatten(delta_flat)

#         new_state = {
#             'lambda': jax.lax.cond(accepted, lambda: trial_lambda * self.decrease_factor, lambda: trial_lambda,
#             ),
#         }
        
#         metrics = {
#             'loss': current_loss,
#             'singular_values': S,
#             'search_direction_norm': jnp.linalg.norm(final_lambda),
#             'lambda': final_lambda,
#             'trial_count': count
#         }

#         return current_loss, updates, new_state, metrics
    
    

class MinLM:
    def __init__(
        self,
        init_lambda=1e3,
        increase_factor: float = 1.05,
        max_increase: float = 2.0,
        plateau_factor: float= 1.05,
    ):
        self.init_lambda = init_lambda
        self.increase_factor = increase_factor
        self.max_increase = max_increase
        self.plateau_factor = plateau_factor

    def init(self, params: eqx.Module):
        return {
            'lambda': jnp.asarray(self.init_lambda),
        }

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        current_lambda = state['lambda']
        
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
        
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        def cond_fun(carry):
            _, accepted, trial_lambda, _, _ = carry
            return (trial_lambda < self.max_increase * current_lambda) & (jnp.logical_not(accepted))

        def body_fun(carry):
            count, _, trial_lambda, previous_loss, previous_plateau = carry
            
            delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
            trial_params = unflatten(params_flat + delta_flat)
            rebuilt_trial_model = eqx.combine(trial_params, static)
            trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
            trial_loss = jnp.mean(trial_residuals**2)

            current_plateau = (self.plateau_factor * trial_loss > previous_loss)

            accepted = (trial_loss <= current_loss) & (previous_plateau) & (current_plateau)

            trial_lambda_new = jax.lax.cond(
                accepted,
                lambda: trial_lambda,
                lambda: trial_lambda * self.increase_factor
            )
            
            return count + 1, accepted, trial_lambda_new, trial_loss, current_plateau

        init_carry = (0, False, current_lambda, current_loss, False)
        count, accepted, trial_lambda, final_loss, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
        final_lambda = jax.lax.cond(accepted, lambda: trial_lambda, lambda: trial_lambda)
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g

        updates = unflatten(delta_flat)

        new_state = {
            'lambda': jax.lax.cond(accepted, lambda: trial_lambda * 0.5, lambda: trial_lambda,
            ),
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
            'trial_count': count
        }

        return current_loss, updates, new_state, metrics
    
    
    


class TernaryLM:
    def __init__(
        self,
        max_iter: int = 30,
    ):
        self.max_iter = max_iter

    def init(self, params: eqx.Module):
        return {
        }

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
    
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        ### TERNARY SEARCH
        
        def get_trial_loss_log(log_lambda):
            """Computes the squared residual loss for a given log_lambda."""
            trial_lambda = jnp.exp(log_lambda)
            
            # Levenberg-Marquardt update direction
            delta_flat = -Vt.T @ jnp.diag(S / (S**2 + trial_lambda)) @ g
            
            # Evaluate the loss with the proposed step
            trial_params = unflatten(params_flat + delta_flat)
            rebuilt_trial_model = eqx.combine(trial_params, static)
            trial_residuals = jax.vmap(partial(residual_fn, rebuilt_trial_model))(collocation_points)
            trial_loss = jnp.mean(trial_residuals**2)
            
            return trial_loss

        S_sq = S**2
        log_lambda_min = jnp.log(jnp.median(S_sq) + 1e-12)
        log_lambda_max = jnp.log(jnp.max(S_sq) + 1e-12)

        def ternary_search_body(i, log_interval):
            """Performs one iteration of ternary search."""
            log_a, log_b = log_interval
            
            # Calculate two intermediate points
            m1 = log_a + (log_b - log_a) / 3
            m2 = log_b - (log_b - log_a) / 3
            
            loss1 = get_trial_loss_log(m1)
            loss2 = get_trial_loss_log(m2)
            
            # Narrow the interval based on which point yielded a lower loss
            new_log_a, new_log_b = jax.lax.cond(
                loss1 < loss2,
                lambda: (log_a, m2), # Keep the left-side interval
                lambda: (m1, log_b)  # Keep the right-side interval
            )
            return (new_log_a, new_log_b)
     
        init_interval = (log_lambda_min, log_lambda_max)
        final_log_a, final_log_b = jax.lax.fori_loop(0, self.max_iter, ternary_search_body, init_interval)

        optimal_log_lambda = (final_log_a + final_log_b) / 2
        optimal_lambda = jnp.exp(optimal_log_lambda)
        final_loss = get_trial_loss_log(optimal_log_lambda)
        
        final_lambda = jax.lax.cond(final_loss < current_loss * 0.9, lambda: optimal_lambda, lambda: jnp.mean(S_sq))
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g
        updates = unflatten(delta_flat)

        new_state = {
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
        }

        return current_loss, updates, new_state, metrics



class MeanLM:
    def __init__(
        self,
    ):
        pass

    def init(self, params: eqx.Module):
        return {
        }

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
    
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        S_sq = S**2
        
        final_lambda = jnp.mean(S_sq)
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g
        updates = unflatten(delta_flat)

        new_state = {
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
        }

        return current_loss, updates, new_state, metrics
    


class MedianLM:
    def __init__(
        self,
    ):
        pass
    
    def init(self, params: eqx.Module):
        return {
        }

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        params_flat, unflatten = ravel_pytree(params)
        
        J, residuals = compute_full_J_and_R(
            params_flat, static, unflatten, residual_fn, collocation_points
        )
    
        current_loss = jnp.mean(residuals**2)

        U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
        g = U.T @ residuals
        
        S_sq = S**2
        
        final_lambda = jnp.median(S_sq)
        
        delta_flat = -Vt.T @ jnp.diag(S / (S**2 + final_lambda)) @ g
        updates = unflatten(delta_flat)

        new_state = {
        }
        
        metrics = {
            'loss': current_loss,
            'singular_values': S,
            'search_direction_norm': jnp.linalg.norm(final_lambda),
            'lambda': final_lambda,
        }

        return current_loss, updates, new_state, metrics
    

import jaxopt


class JaxoptWrapper:
    """
    A refactored wrapper that is initialized with all static problem information,
    leading to a cleaner and more direct API.
    """
    def __init__(
        self, 
        residual_fn,
        static,
        solver="bfgs",
        **solver_options
    ):
        """
        Initializes the wrapper with the static parts of the problem definition.

        Args:
            residual_fn: The function that computes the problem's residual.
            static: The static (non-trainable) part of the equinox model.
            solver: An instance of a jaxopt solver. The `fun` argument will be set here.
        """
        self.residual_fn = residual_fn
        self.static = static
        
        # Define the REAL loss function once, here.
        # It takes parameters and the collocation points as arguments.
        def loss_fn(p, collocation_points):
            model = eqx.combine(p, self.static)
            residuals = jax.vmap(lambda point: self.residual_fn(model, point))(collocation_points)
            return jnp.mean(residuals**2)

        solver_name = solver.lower()
        if solver_name == "lbfgs":
            self.solver = jaxopt.LBFGS(fun=loss_fn, **solver_options)
        elif solver_name == "bfgs":
            self.solver = jaxopt.BFGS(fun=loss_fn, **solver_options)
        elif solver_name == "scipy-bfgs":
            self.solver = jaxopt.ScipyMinimize(fun=loss_fn, method="BFGS", **solver_options)
        elif solver_name == "scipy-lbfgs":
            self.solver = jaxopt.ScipyMinimize(fun=loss_fn, method="L-BFGS-B", **solver_options)
        elif solver_name == "scipy-cg":
            self.solver = jaxopt.ScipyMinimize(fun=loss_fn, method="CG", **solver_options)
        
        if "scipy" in solver_name:
            self.is_scipy_solver = True
            
        # self.solver = dataclasses.replace(solver, fun=loss_fn, **solver_options)


    def init(self, params, initial_collocation_points):
        if self.is_scipy_solver:
            # Scipy solvers don't have a state to initialize.
            return None
        
        return self.solver.init_state(params, collocation_points=initial_collocation_points)

    def update(
        self,
        params,
        static,
        state,
        residual_fn,
        collocation_points
    ):
        
        if self.is_scipy_solver:
            results = self.solver.run(params, collocation_points=collocation_points)
            new_params = results.params
            current_loss = results.state.fun_val
            new_state = None
        else:
            # Standard update for JAX-native solvers
            new_params, new_state = self.solver.update(
                params, 
                state=state, 
                collocation_points=collocation_points
            )
            current_loss = new_state.error
        
        updates = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)

        current_loss = new_state.error
        metrics = {"loss": current_loss}

        return current_loss, updates, new_state, metrics




class BFGS:
    """
    BFGS optimiser with a robust Wolfe line search and Nocedal & Wright Hessian scaling.
    This version includes robust failure handling for the line search and Hessian update.
    """

    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls_steps: int = 20,
        alpha_init: float = 1.0,
    ):
        """
        Initialises the BFGS optimiser.
        
        Args:
            c1 (float): Armijo condition constant.
            c2 (float): Curvature condition constant.
            max_ls_steps (int): Maximum steps for line search.
            alpha_init (float): Initial step size guess.
        """
        self.c1 = c1
        self.c2 = c2
        self.max_ls_steps = max_ls_steps
        self.alpha_init = alpha_init

    def init(self, params: eqx.Module):
        """
        Initialises the optimizer state.
        
        Returns:
            dict: The initial state dictionary.
        """
        params_flat, _ = ravel_pytree(params)
        num_params = params_flat.size
        return {
            # H_inv is the inverse Hessian approximation
            'H_inv': jnp.eye(num_params, dtype=params_flat.dtype),
            # g_prev is the gradient from the previous step
            'g_prev': jnp.zeros_like(params_flat),
            # is_first_step is a flag to apply initial Hessian scaling
            'is_first_step': jnp.array(True, dtype=bool)
        }

    def _line_search(
        self,
        loss_fn: Callable,
        params: eqx.Module,
        current_loss: jax.Array,
        p_k: jax.Array,
        unflatten: Callable,
        params_flat: jax.Array,
        current_grad_flat: jax.Array,
    ):
        """
        Performs a robust Wolfe line search to find an appropriate step size.
        """
        directional_derivative = jnp.dot(current_grad_flat, p_k)

        # Loop body for the while_loop
        def loop_body(state):
            alpha, ls_steps, _, _, _, _ = state
            
            # Update flat parameters with the current trial step size alpha
            temp_new_params_flat = params_flat + alpha * p_k
            new_params_tree = unflatten(temp_new_params_flat)
            
            # Evaluate loss and gradient at the new point
            new_loss, new_grad = jax.value_and_grad(loss_fn)(new_params_tree)
            new_grad_flat, _ = ravel_pytree(new_grad)

            # Check Wolfe conditions
            armijo_cond = new_loss <= current_loss + self.c1 * alpha * directional_derivative
            curvature_cond = jnp.dot(new_grad_flat, p_k) >= self.c2 * directional_derivative
            
            conditions_met = armijo_cond & curvature_cond
            
            # If conditions are met, alpha remains the same, which terminates the loop.
            # Otherwise, backtrack by halving alpha.
            next_alpha = jax.lax.cond(conditions_met, lambda: alpha, lambda: alpha * 0.5)
            
            return (next_alpha, ls_steps + 1, new_loss, new_params_tree, new_grad, conditions_met)

        # Initial loop state
        initial_state = (self.alpha_init, 0, current_loss, params, None, False)
        
        # While loop to perform line search
        final_alpha, final_ls_steps, final_loss, final_params, final_grad, conditions_met = jax.lax.while_loop(
            lambda state: (state[1] < self.max_ls_steps) & ~state[5], # Loop while steps < max and conditions not met
            loop_body,
            initial_state
        )
        
        is_successful = conditions_met
        return final_loss, final_params, final_grad, final_alpha, final_ls_steps, is_successful

    def update(
        self,
        params: eqx.Module,
        static,
        state: optax.OptState,
        residual_fn: Callable[[eqx.Module], jax.Array],
        collocation_points
    ):
        """
        Updates the parameters using the BFGS algorithm.
        """
        def loss_fn(p):
            model = eqx.combine(p, static)
            trial_residuals = jax.vmap(partial(residual_fn, model))(collocation_points)
            return jnp.mean(trial_residuals**2)
        
        H_inv = state['H_inv']
        g_prev = state['g_prev']
        is_first_step = state['is_first_step']

        (current_loss, current_grad) = jax.value_and_grad(loss_fn)(params)
        params_flat, unflatten = ravel_pytree(params)
        current_grad_flat, _ = ravel_pytree(current_grad)
        
        # Compute the search direction: p_k = -H_inv * grad
        p_k = -jnp.dot(H_inv, current_grad_flat)
        
        # Perform Wolfe line search
        new_loss, new_params_tree, new_grad, alpha, ls_steps, is_successful_step = self._line_search(
            loss_fn, params, current_loss, p_k, unflatten, params_flat, current_grad_flat
        )
        
        new_params_flat, _ = ravel_pytree(new_params_tree)
        new_grad_flat, _ = ravel_pytree(new_grad)

        def _successful_step_fn():
            # Calculate s_k and y_k from the line search results
            s_k = new_params_flat - params_flat
            y_k = new_grad_flat - current_grad_flat
            
            s_k_dot_y_k = jnp.dot(s_k, y_k)
            rho_k = 1.0 / s_k_dot_y_k
            
            # Nocedal & Wright initial scaling for the first step
            H_inv_maybe_scaled = jax.lax.cond(
                is_first_step,
                lambda: H_inv * (s_k_dot_y_k / jnp.dot(y_k, y_k)),
                lambda: H_inv
            )
            
            # Standard BFGS update formula
            term1 = jnp.eye(H_inv.shape[0], dtype=H_inv.dtype) - rho_k * jnp.outer(s_k, y_k)
            term2 = jnp.eye(H_inv.shape[0], dtype=H_inv.dtype) - rho_k * jnp.outer(y_k, s_k)
            H_inv_updated = jnp.dot(jnp.dot(term1, H_inv_maybe_scaled), term2) + rho_k * jnp.outer(s_k, s_k)
            
            # Conditionally apply the update only if curvature is positive
            final_H_inv = jax.lax.cond(
                s_k_dot_y_k > 0,
                lambda: H_inv_updated,
                lambda: H_inv # Keep old H_inv if curvature is not positive
            )
            
            updates_flat = s_k
            
            new_state = {
                'H_inv': final_H_inv,
                'g_prev': new_grad_flat,
                'is_first_step': jnp.array(False, dtype=bool)
            }
            return updates_flat, new_state

        def _failed_step_fn():
            # Line search failed; reset Hessian and take no step
            updates_flat = jnp.zeros_like(params_flat)
            new_state = {
                'H_inv': jnp.eye(H_inv.shape[0], dtype=H_inv.dtype),
                'g_prev': current_grad_flat, # Keep the old gradient
                'is_first_step': jnp.array(True, dtype=bool) # Reset for scaling on next attempt
            }
            return updates_flat, new_state

        updates_flat, new_state = jax.lax.cond(
            is_successful_step,
            _successful_step_fn,
            _failed_step_fn
        )
        
        updates = unflatten(updates_flat)

        metrics = {
            'loss': jax.lax.cond(is_successful_step, lambda: new_loss, lambda: current_loss),
            'alpha': alpha,
            'line_search_steps': ls_steps,
            'is_successful_step': is_successful_step,
            'grad_norm': jnp.linalg.norm(current_grad_flat),
        }

        return current_loss, updates, new_state, metrics