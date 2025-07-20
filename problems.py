import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx
from functools import partial
from typing import Callable
from scipy.stats import qmc

from traditional_solutions.burgers import burgers_viscous_time_exact1

class Burgers:
    """
    Defines the problem for the viscous Burgers' equation.

    PDE:         u_t + u * u_x = nu * u_xx
    Domain:      x ∈ [-1, 1], t ∈ [0, 1]
    BC (Boundary): u(-1, t) = u(1, t) = 0
    IC (Initial):  u(x, 0) = -sin(πx)
    """
    # Problem parameters
    nu: float
    xmin: float = -1.0
    xmax: float = 1.0
    tmin: float = 0.0
    tmax: float = 1.0

    def __init__(self, nu: float = 0.01 / jnp.pi):
        """Initializes the problem with a given viscosity."""
        self.nu = nu

    def exact_solution(self, nx: int = 128, nt: int = 64):
        """
        Generates the exact solution, formatted for PINN training data.

        Args:
            nx: Number of spatial grid points.
            nt: Number of temporal grid points.

        Returns:
            A tuple containing:
            - coords: A `(nx*nt, 2)` array of (x, t) coordinates.
            - u: A `(nx*nt, 1)` array of the corresponding solution u(x, t).
        """
        vx = np.linspace(self.xmin, self.xmax, nx)
        vt = np.linspace(self.tmin, self.tmax, nt)
        
        u_grid = burgers_viscous_time_exact1(self.nu, nx, vx, nt, vt)
        
        X, T = np.meshgrid(vx, vt, indexing='ij')
        u = u_grid.flatten()[:, None]
        
        return X, T, u.reshape((nx, nt))

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh):
            if layer_sizes is None:
                layer_sizes = [2, 10, 10, 10, 10, 1]  # Default network architecture
            
            self.activation = activation
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))

        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            """
            Forward pass of the PINN, with hard-constraints for BC/IC.
            
            This method uses an ansatz to enforce the conditions:
            u(x,t) = (1-g(t))*IC(x) + B(x)*g(t)*NN(x,t)
            where g(t) is a smooth function that is 0 at t=0 and 1 otherwise,
            and B(x) is a function that is 0 at the spatial boundaries.
            """
            x, t = xt[0], xt[1]
            
            nn_input = xt
            for i, layer in enumerate(self.layers):
                nn_input = layer(nn_input)
                if i < len(self.layers) - 1:
                    nn_input = self.activation(nn_input)
            nn_output = nn_input[0]

            sd = 0.05  # Steepness for tanh approximations of step functions
            
            time_factor = jax.nn.tanh(t / sd)
            boundary_factor = jax.nn.tanh((x + 1) / sd) * jax.nn.tanh((1 - x) / sd)
            initial_condition = -jnp.sin(jnp.pi * x)

            u = (1 - time_factor) * initial_condition + boundary_factor * time_factor * nn_output
            
            return u

    # --- Derivative and Residual Calculation Utilities ---
    # These static functions compute derivatives required for the PDE residual.

    # Gradient of u w.r.t. xt=(x, t) -> returns [u_x, u_t]
    _u_grad_fn = jax.grad(lambda model, xt: model(xt), argnums=1)
    
    # Gradient of u_x w.r.t. xt -> returns [u_xx, u_xt]
    _u_xx_fn = jax.grad(lambda model, xt: Burgers._u_grad_fn(model, xt)[0], argnums=1)

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def physics_residual(residual_fn: Callable, model: FCN, xt: jnp.ndarray, nu) -> jnp.ndarray:
        # Compute derivatives at the single point `xt`
        u = model(xt)
        u_x, u_t = Burgers._u_grad_fn(model, xt)
        u_xx, _ = Burgers._u_xx_fn(model, xt)
        
        return residual_fn(u, u_x, u_t, u_xx, nu)
    
    @staticmethod
    def residual_fn(u: jnp.ndarray, u_x: jnp.ndarray, u_t: jnp.ndarray, u_xx: jnp.ndarray, nu: float) -> jnp.ndarray:
        """Calculates the physics residual of the Burgers' equation: f = u_t + u*u_x - nu*u_xx."""
        return u_t + u * u_x - nu * u_xx
    
    def get_collocation_points(self, n, key):
        sampler = qmc.LatinHypercube(d=2, seed=key)
        samples = sampler.random(n=n)
        l_bounds = [self.xmin, self.tmin]
        u_bounds = [self.xmax, self.tmax]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        return jnp.array(scaled_samples)