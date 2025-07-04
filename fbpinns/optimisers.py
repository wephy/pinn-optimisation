import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map, tree_unflatten

from functools import partial

from typing import NamedTuple, Callable, Any, Optional
from typing import Any, Callable, NamedTuple

import optax


class gaussnewton_lsqr:
    """
    A JAX-based implementation of the Gauss-Newton algorithm using jax.numpy.linalg.lsqr
    to solve the linear subproblem at each iteration.
    """

    def __init__(self, damping_factor: float = 1e-7, lsqr_iter_lim: int = -1):
        """
        Initializes the Gauss-Newton optimizer.

        Args:
            damping_factor (float, optional): Regularization factor for LSQR. Defaults to 1e-8.
            lsqr_iter_lim (int, optional): Iteration limit for LSQR. Defaults to -1 (auto).
        """
        self.damping_factor = damping_factor
        self.lsqr_iter_lim = lsqr_iter_lim

    def init(self, params: Any) -> None:
        """Initializes the optimizer state (stateless for this implementation)."""
        return None

    def update(
        self,
        state: Any,  # The state from the base optimizer
        params: optax.Params,
        residual_fn: Callable[[optax.Params], float]
    ) -> tuple[float, Any, Any]:
        """
        Performs one step of the Gauss-Newton optimization.
        This optimizer requires `residual_fn` to be passed.
        """
        if residual_fn is None:
            raise ValueError(
                "GaussNewtonLSQR requires a 'residual_fn' to be provided to the update method.")

        residuals, jacobian_vjp = jax.vjp(residual_fn, params)
        flat_params, tree_def = jax.tree_util.tree_flatten(params)
        if not flat_params:
            raise ValueError("The model has no trainable parameters.")

        def jvp(v_flat):
            v_tree = tree_unflatten(tree_def, v_flat)
            _, tangent = jax.jvp(residual_fn, (params,), (v_tree,))
            return tangent

        def vjp_lin(v):
            param_grads_tree = jacobian_vjp(v)[0]
            flat_grad, _ = jax.tree_util.tree_flatten(param_grads_tree)
            return jnp.concatenate([g.ravel() for g in flat_grad])

        num_vars = sum(p.size for p in flat_params)
        J_operator = jax.scipy.sparse.linalg.LinearOperator(
            shape=(residuals.shape[0], num_vars),
            matvec=jvp,
            rmatvec=vjp_lin
        )

        step_flat, *_ = jnp.linalg.lsqr(
            J_operator,
            -residuals,
            damp=self.damping_factor,
            iter_lim=self.lsqr_iter_lim if self.lsqr_iter_lim != -1 else 100
        )

        step_tree = tree_unflatten(tree_def, step_flat)
        updates = tree_map(lambda x: -x, step_tree)
        loss = 0.5 * jnp.sum(residuals**2)
        return loss, updates, None


class adam:
    """
    Wrapper for optax.adam
    """

    def __init__(self, **kwargs):
        """
        Initializes the wrapper.

        Args:
          **kwargs: Keyword arguments to be passed, e.g. `learning_rate'
        """

        self.base_optimizer = optax.adam(**kwargs)
        self.init = self.base_optimizer.init

    def update(
        self,
        state: Any,  # The state from the base optimizer
        params: optax.Params,
        residual_fn: Callable[[optax.Params], float]
    ) -> tuple[float, Any, Any]:
        """
        Performs one full optimization step
        """
        def internal_loss_fn(p):
            residuals = residual_fn(p)
            return 0.5 * jnp.sum(residuals**2)

        loss, grads = jax.value_and_grad(internal_loss_fn)(params)

        updates, new_optimizer_state = self.base_optimizer.update(
            grads, state, params)
        return loss, updates, new_optimizer_state


class lbfgs:
    """
    Wrapper for optax.lbfgs
    """

    def __init__(self, **kwargs):
        """
        Initializes the wrapper.

        Args:
          **kwargs: Keyword arguments to be passed to the optax.lbfgs optimizer,
                    such as `tol`, `max_linesearch_steps`, `use_zoom_linesearch`, etc.
        """

        self.base_optimizer = optax.lbfgs(**kwargs)
        self.init = self.base_optimizer.init

    def update(
        self,
        state: Any,  # The state from the base optimizer
        params: optax.Params,
        residual_fn: Callable[[optax.Params], float]
    ) -> tuple[float, optax.Updates, Any]:
        """
        Performs one full optimization step
        """
        def internal_loss_fn(p):
            residuals = residual_fn(p)
            return 0.5 * jnp.sum(residuals**2)
        loss, grads = jax.value_and_grad(internal_loss_fn)(params)

        updates, new_optimizer_state = self.base_optimizer.update(
            grads, state, params, value=loss, grad=grads, value_fn=internal_loss_fn
        )

        return loss, updates, new_optimizer_state
