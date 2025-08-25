import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx
from functools import partial
from typing import Callable, List, Tuple
from scipy.stats import qmc, norm
from scipy.integrate import solve_ivp

from traditional_solutions.burgers import burgers_viscous_time_exact1
from traditional_solutions.wave import wave_1d_time_exact
from traditional_solutions.poisson1d import poisson_1d_exact

# from optimisers import resample_rad

class Burgers:
    """
    Defines the problem for the viscous Burgers' equation.

    PDE:         u_t + u * u_x = nu * u_xx
    Domain:      x ∈ [-1, 1], t ∈ [0, 1]
    BC (Boundary): u(-1, t) = u(1, t) = 0
    IC (Initial):  u(x, 0) = -sin(πx)
    """
    # Problem parameters
    nu: float = 0.01 / jnp.pi
    xmin: float = -1.0
    xmax: float = 1.0
    tmin: float = 0.0
    tmax: float = 1.0

    def __init__(self):
        """Initializes the problem with a given viscosity."""
        # self.nu = nu
        pass
    
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
        
        u_grid = burgers_viscous_time_exact1(Burgers.nu, nx, vx, nt, vt)
        
        X, T = np.meshgrid(vx, vt, indexing='ij')
        u = u_grid.flatten()[:, None]
        
        return X, T, u.reshape((nx, nt))

    class FCN(eqx.Module):
        # 2. Declare all fields at the class level
        weights: list[jax.Array]
        biases: list[jax.Array]
        
        # Mark non-trainable attributes as static
        activation: Callable = eqx.field(static=True)
        """A simple fully connected neural network (MLP) using Equinox."""

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))
            

        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            """
            Forward pass of the PINN, with hard-constraints for BC/IC.
            """
            x, t = xt[0], xt[1]
            
            nn_input = xt
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]

            sd = 0.075
            
            time_factor = jax.nn.tanh(t / sd)
            boundary_factor = jax.nn.tanh((x + 1) / sd) * jax.nn.tanh((1 - x) / sd)
            initial_condition = -jnp.sin(jnp.pi * x)

            u = initial_condition * (1 - time_factor) + boundary_factor * time_factor * nn_output
            
            return u

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape)

    # --- Derivative and Residual Calculation Utilities ---
    _u_grad_fn = jax.grad(lambda model, xt: model(xt), argnums=1)
    _u_xx_fn = jax.grad(lambda model, xt: Burgers._u_grad_fn(model, xt)[0], argnums=1)

    @staticmethod
    def physics_residual(model: FCN, xt: jnp.ndarray) -> jnp.ndarray:
        # Compute derivatives at the single point `xt`
        u = model(xt)
        u_x, u_t = Burgers._u_grad_fn(model, xt)
        u_xx, _ = Burgers._u_xx_fn(model, xt)
        
        return u_t + u * u_x - (0.01 / jnp.pi) * u_xx

    def get_candidate_points(self, n_candidates, seed):
        """Generates a large, uniform pool of candidate points for resampling."""
        key = jax.random.PRNGKey(seed)
        # Generate candidates uniformly across the domain
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.xmin, maxval=self.xmax)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.tmin, maxval=self.tmax)
        
        return jnp.hstack([x_candidates, t_candidates])









class AllenCahnKarniadakis:
    """
    Defines the problem for the Allen-Cahn equation.

    PDE:         u_t - 10^{-4} * u_xx + 5 * u^3 - 5 * u = 0
    Domain:      x ∈ [-1, 1], t ∈ [0, 1]
    BC (Periodic): u(t, -1) = u(t, 1), u_x(t, -1) = u_x(t, 1)
    IC (Initial):  u(0, x) = x^2 * cos(πx)
    """
    # Problem parameters
    nu: float = 1e-4
    xmin: float = -1.0
    xmax: float = 1.0
    tmin: float = 0.0
    tmax: float = 1.0

    def __init__(self):
        """Initializes the problem with a given viscosity."""
        pass
    
    def numerical_solver(self, nx: int = 128, nt: int = 64):
        """
        Generates a numerical solution using a more accurate
        4th-order Runge-Kutta (RK4) finite difference method.

        Args:
            nx: Number of spatial grid points.
            nt: Number of temporal grid points.

        Returns:
            A tuple containing:
            - X: A `(nx, nt)` array of spatial coordinates.
            - T: A `(nx, nt)` array of temporal coordinates.
            - u_grid: A `(nx, nt)` array of the numerical solution u(x, t).
        """
        # Create a mesh grid
        x_np = np.linspace(self.xmin, self.xmax, nx)
        t_np = np.linspace(self.tmin, self.tmax, nt)
        X, T = np.meshgrid(x_np, t_np, indexing='ij')

        # Calculate step sizes
        dx = (self.xmax - self.xmin) / (nx - 1)
        dt = (self.tmax - self.tmin) / (nt - 1)

        # Initialize the solution grid
        u_grid = np.zeros((nx, nt))
        
        # Apply the initial condition
        u_grid[:, 0] = x_np**2 * np.sin(2 * np.pi * x_np)

        # Define the right-hand side function for the PDE
        def rhs(u_vec):
            # The second derivative u_xx using a finite difference approximation
            u_xx = (np.roll(u_vec, -1) - 2 * u_vec + np.roll(u_vec, 1)) / dx**2
            
            # The reaction term
            reaction_term = -5 * u_vec**3 + 5 * u_vec

            # The full right-hand side of the PDE
            return self.nu * u_xx + reaction_term

        # Time-stepping using the RK4 method
        for n in range(nt - 1):
            u_n = u_grid[:, n]
            
            # RK4 steps
            k1 = dt * rhs(u_n)
            k2 = dt * rhs(u_n + 0.5 * k1)
            k3 = dt * rhs(u_n + 0.5 * k2)
            k4 = dt * rhs(u_n + k3)
            
            # Update the solution for the next time step
            u_grid[:, n+1] = u_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
        return X, T, u_grid
    
    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        # Declare all fields at the class level.
        weights: list[jax.Array]
        biases: list[jax.Array]
        
        # Mark non-trainable attributes as static.
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int], activation: Callable = jax.nn.tanh):
            """Initializes the network with a given architecture and activation function."""
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))
        
        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            x, t = xt[0], xt[1]

            initial_condition = x**2 * jnp.sin(2 * jnp.pi * x)

            periodic_features = jnp.hstack([
                jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x),
            ])
            
            nn_input = jnp.hstack([t, periodic_features])

            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            
            # The network output is the "correction" term.
            nn_output = nn_input[0]
            
            # sd = 0.05
                        
            u = initial_condition + nn_output * jax.nn.tanh(t / 0.01) * jax.nn.tanh((x + 1) / 0.001) * jax.nn.tanh((1 - x) / 0.001)
            
            return u

        def predict(self, X: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the solution u(x,t) for a given grid of (x,t) points.
            """
            @partial(jax.vmap)
            def _model_prediction(points: jnp.ndarray) -> jnp.ndarray:
                return self.__call__(points)

            # Flatten the grid into a list of (x,t) points for vectorized prediction.
            test_points = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            
            # Reshape the output to match the original grid shape.
            return _model_prediction(test_points).reshape(X.shape)


    @staticmethod
    def physics_residual(model: FCN, xt: jnp.ndarray) -> jnp.ndarray:
        u = model(xt)

        u_t = jax.grad(lambda t: model(jnp.array([xt[0], t])), argnums=0)(xt[1])
        
        def grad_u_x(x_val):
            return jax.grad(lambda x_in: model(jnp.array([x_in, xt[1]])), argnums=0)(x_val)
        
        u_xx = jax.grad(grad_u_x, argnums=0)(xt[0])

        return u_t - 1e-4 * u_xx + 5 * (u**3 - u)

    def get_candidate_points(self, n_candidates: int, seed: int) -> jnp.ndarray:
        """Generates a large, uniform pool of candidate points for sampling."""
        key = jax.random.PRNGKey(seed)
        # Generate candidates uniformly across the spatial and temporal domains.
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.xmin, maxval=self.xmax)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.tmin, maxval=self.tmax)
        
        return jnp.hstack([x_candidates, t_candidates])


























class AllenCahn:
    """
    Defines the problem for the Allen-Cahn equation.

    PDE:         u_t - 10^{-4} * u_xx + 5 * u^3 - 5 * u = 0
    Domain:      x ∈ [-1, 1], t ∈ [0, 1]
    BC (Periodic): u(t, -1) = u(t, 1), u_x(t, -1) = u_x(t, 1)
    IC (Initial):  u(0, x) = x^2 * cos(πx)
    """
    # Problem parameters
    nu: float = 1e-4
    xmin: float = -1.0
    xmax: float = 1.0
    tmin: float = 0.0
    tmax: float = 1.0

    def __init__(self):
        """Initializes the problem with a given viscosity."""
        pass
    
    def numerical_solver(self, nx: int = 128, nt: int = 64):
        """
        Generates a numerical solution using a more accurate
        4th-order Runge-Kutta (RK4) finite difference method.

        Args:
            nx: Number of spatial grid points.
            nt: Number of temporal grid points.

        Returns:
            A tuple containing:
            - X: A `(nx, nt)` array of spatial coordinates.
            - T: A `(nx, nt)` array of temporal coordinates.
            - u_grid: A `(nx, nt)` array of the numerical solution u(x, t).
        """
        # Create a mesh grid
        x_np = np.linspace(self.xmin, self.xmax, nx)
        t_np = np.linspace(self.tmin, self.tmax, nt)
        X, T = np.meshgrid(x_np, t_np, indexing='ij')

        # Calculate step sizes
        dx = (self.xmax - self.xmin) / (nx - 1)
        dt = (self.tmax - self.tmin) / (nt - 1)

        # Initialize the solution grid
        u_grid = np.zeros((nx, nt))
        
        # Apply the initial condition
        u_grid[:, 0] = x_np**2 * np.cos(np.pi * x_np)

        # Define the right-hand side function for the PDE
        def rhs(u_vec):
            # The second derivative u_xx using a finite difference approximation
            u_xx = (np.roll(u_vec, -1) - 2 * u_vec + np.roll(u_vec, 1)) / dx**2
            
            # The reaction term
            reaction_term = -5 * u_vec**3 + 5 * u_vec

            # The full right-hand side of the PDE
            return self.nu * u_xx + reaction_term

        # Time-stepping using the RK4 method
        for n in range(nt - 1):
            u_n = u_grid[:, n]
            
            # RK4 steps
            k1 = dt * rhs(u_n)
            k2 = dt * rhs(u_n + 0.5 * k1)
            k3 = dt * rhs(u_n + 0.5 * k2)
            k4 = dt * rhs(u_n + k3)
            
            # Update the solution for the next time step
            u_grid[:, n+1] = u_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
        return X, T, u_grid
    
    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        # Declare all fields at the class level.
        weights: list[jax.Array]
        biases: list[jax.Array]
        
        # Mark non-trainable attributes as static.
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int], activation: Callable = jax.nn.tanh):
            """Initializes the network with a given architecture and activation function."""
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))
        
        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            x, t = xt[0], xt[1]

            initial_condition = x**2 * jnp.cos(jnp.pi * x)

            periodic_features = jnp.hstack([
                jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x),
            ])
            
            nn_input = jnp.hstack([t, periodic_features])

            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            
            # The network output is the "correction" term.
            nn_output = nn_input[0]
            
            # sd = 0.05
                        
            u = initial_condition + nn_output * jax.nn.tanh(t / 0.05) 
            
            return u

        def predict(self, X: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the solution u(x,t) for a given grid of (x,t) points.
            """
            @partial(jax.vmap)
            def _model_prediction(points: jnp.ndarray) -> jnp.ndarray:
                return self.__call__(points)

            # Flatten the grid into a list of (x,t) points for vectorized prediction.
            test_points = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            
            # Reshape the output to match the original grid shape.
            return _model_prediction(test_points).reshape(X.shape)


    @staticmethod
    def physics_residual(model: FCN, xt: jnp.ndarray) -> jnp.ndarray:
        u = model(xt)

        u_t = jax.grad(lambda t: model(jnp.array([xt[0], t])), argnums=0)(xt[1])
        
        def grad_u_x(x_val):
            return jax.grad(lambda x_in: model(jnp.array([x_in, xt[1]])), argnums=0)(x_val)
        
        u_xx = jax.grad(grad_u_x, argnums=0)(xt[0])

        return u_t - 1e-4 * u_xx + 5 * (u**3 - u)

    def get_candidate_points(self, n_candidates: int, seed: int) -> jnp.ndarray:
        """Generates a large, uniform pool of candidate points for sampling."""
        key = jax.random.PRNGKey(seed)
        # Generate candidates uniformly across the spatial and temporal domains.
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.xmin, maxval=self.xmax)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.tmin, maxval=self.tmax)
        
        return jnp.hstack([x_candidates, t_candidates])
























class Wave:
    """
    Defines the problem for the 1D wave equation.

    PDE:           u_tt = c^2 * u_xx
    Domain:        x ∈ [-1, 1], t ∈ [0, 1]
    BC (Boundary): u(-1, t) = u(1, t) = 0
    IC (Initial):  u(x, 0) = -sin(πx), u_t(x, 0) = 0
    """
    # Problem parameters
    c: float
    xmin: float = -1.0
    xmax: float = 1.0
    tmin: float = 0.0
    tmax: float = 1.0

    def __init__(self, c: float = 1.0):
        """Initializes the problem with a given wave speed."""
        self.c = c

    def exact_solution(self, nx: int = 128, nt: int = 64):
        """
        Generates the exact solution, formatted for PINN training data.

        Args:
            nx: Number of spatial grid points.
            nt: Number of temporal grid points.

        Returns:
            A tuple containing:
            - X: A `(nx, nt)` meshgrid of x coordinates.
            - T: A `(nx, nt)` meshgrid of t coordinates.
            - u: A `(nx, nt)` array of the corresponding solution u(x, t).
        """
        vx = np.linspace(self.xmin, self.xmax, nx)
        vt = np.linspace(self.tmin, self.tmax, nt)
        
        u_grid = wave_1d_time_exact(self.c, nx, vx, nt, vt)
        
        X, T = np.meshgrid(vx, vt, indexing='ij')
        return X, T, u_grid

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


            time_factor = jax.nn.tanh(t / 0.05)
            boundary_factor = jax.nn.tanh((x + 1) / 0.1) * jax.nn.tanh((1 - x) / 0.1)
            initial_condition = -jnp.sin(jnp.pi * x)

            u = (1 - time_factor) * initial_condition + boundary_factor * time_factor * nn_output
            
            return u

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape)

    # --- Derivative and Residual Calculation Utilities ---
    # These static functions compute derivatives required for the PDE residual.

    # Gradient of u w.r.t. xt=(x, t) -> returns [u_x, u_t]
    _u_hessian_fn = jax.hessian(lambda model, xt: model(xt), argnums=1)

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def physics_residual(residual_fn: Callable, model: FCN, xt: jnp.ndarray, c: float) -> jnp.ndarray:
        # Compute derivatives at the single point `xt`
        u_hessian = Wave._u_hessian_fn(model, xt)
        u_xx = u_hessian[0, 0]
        u_tt = u_hessian[1, 1]
        
        return residual_fn(u_xx, u_tt, c)
    
    @staticmethod
    def residual_fn(u_xx: jnp.ndarray, u_tt: jnp.ndarray, c: float) -> jnp.ndarray:
        """Calculates the physics residual of the Wave equation: f = u_tt - c^2 * u_xx."""
        return u_tt - c**2 * u_xx
    
    def get_collocation_points(self, n, seed):
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n=n)
        l_bounds = [self.xmin, self.tmin]
        u_bounds = [self.xmax, self.tmax]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        return jnp.array(scaled_samples)


class Poisson1D:
    """
    Defines the problem for the 1D Poisson.

    PDE:           u_xx = (4x^3 - 6x)exp(-x^2)
    BC:            u(0) = 0, u(1) = exp(-1)
    """
    # Problem parameters
    xmin: float = 0.0
    xmax: float = 1.0

    def __init__(self):
        """Initializes the problem with a given wave speed."""
        pass

    def exact_solution(self, nx: int = 128):
        """Exact solution on mesh"""
        vx = np.linspace(self.xmin, self.xmax, nx)
        
        u_grid = poisson_1d_exact(nx, vx)

        return vx, u_grid

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh):
            if layer_sizes is None:
                layer_sizes = [1, 10, 10, 10, 1]  # Default network architecture
            
            self.activation = activation
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """
            Forward pass of the PINN, with hard-constraints for BC/IC.
            """
            nn_input = jnp.reshape(x, (1,))
            for i, layer in enumerate(self.layers):
                nn_input = layer(nn_input)
                if i < len(self.layers) - 1:
                    nn_input = self.activation(nn_input)
            
            nn_output = nn_input[0]

            sd = 0.1

            boundary_factor = jax.nn.tanh(x / sd) * jax.nn.tanh((1 - x) / sd)
            initial_condition = jnp.exp(-1) * (1 - jax.nn.tanh((1 - x) / sd))

            u = initial_condition + boundary_factor * nn_output
            
            return u

        def predict(self, X):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = X
            return _model_prediction(test_points).reshape(X.shape)

    _u_xx_fn = jax.grad(jax.grad(lambda model, x: model(x), argnums=1), argnums=1)

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0))
    def physics_residual(residual_fn: Callable, model: FCN, x: jnp.ndarray) -> jnp.ndarray:
        # Compute derivatives at the single point `xt`
        u_xx = Poisson1D._u_xx_fn(model, x)
        
        return residual_fn(u_xx, x)
    
    @staticmethod
    def residual_fn(u_xx: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Calculates the physics residual."""
        return u_xx - (4 * x**3 - 6 * x) * jnp.exp(-x**2)
    
    def get_collocation_points(self, n, seed):
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        samples = sampler.random(n=n)
        l_bounds = [self.xmin]
        u_bounds = [self.xmax]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        return jnp.array(scaled_samples).squeeze()


class Poisson2D:
    """
    Defines the problem for a 2D Poisson equation.

    PDE:         Δu(x, y) = f(x, y)
    Domain:      (x, y) ∈ (0, 1)²
    BCs:         u(x, 0) = 0
                 ∂u/∂n(0, y) = ∂u/∂n(1, y) = ∂u/∂n(x, 1) = 0
    """
    # Problem parameters
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0

    def __init__(self):
        """Initializes the problem."""
        pass
    
    def exact_solution(self, nx: int = 128, ny: int = 128):
        """Generates the exact analytical solution u(x,y) = x²(x-1)²y(y-1)²."""
        vx = np.linspace(self.xmin, self.xmax, nx)
        vy = np.linspace(self.ymin, self.ymax, ny)
        X, Y = np.meshgrid(vx, vy, indexing='ij')
        
        u_grid = (X**2) * ((X - 1)**2) * Y * ((Y - 1)**2)
        
        return X, Y, u_grid

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh):
            if layer_sizes is None:
                # Input is (x,y), output is u
                layer_sizes = [2, 20, 20, 20, 20, 1]  
            
            self.activation = activation
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))

        def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
            """A standard forward pass. Hard constraints are omitted for simplicity."""
            x, y = xy[0], xy[1]
            
            nn_input = xy
            for i, layer in enumerate(self.layers):
                nn_input = layer(nn_input)
                if i < len(self.layers) - 1:
                    nn_input = self.activation(nn_input)
            nn_output = nn_input[0]

            sd = 0.025
            
            boundary_factor = jax.nn.tanh(x / sd) * jax.nn.tanh((1 - x) / sd) * jax.nn.tanh(y / sd) * jax.nn.tanh((1 - y) / sd)
        
            return nn_output * boundary_factor

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape)


    @staticmethod
    def physics_residual(model: FCN, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the physics residual for the 2D Poisson equation with a
        numerically stable implementation.
        """
        x, y = xy[0], xy[1]

        # Define a function u that takes a coordinate array and returns a scalar
        def u_fn(coords):
            # The .squeeze() is important to ensure the output is a scalar,
            # which jax.hessian expects.
            return model(coords).squeeze()

        # FAST: Compute the Laplacian directly as the trace of the Hessian
        laplacian_u = jnp.trace(jax.hessian(u_fn)(xy))

        # The source term is unchanged
        source_term = 2 * (x**4 * (3*y - 2) + x**3 * (4 - 6*y) +
                           x**2 * (6*y**3 - 12*y**2 + 9*y - 2) -
                           6*x * (y - 1)**2 * y + (y - 1)**2 * y)
                           
        return laplacian_u - source_term

    
    def get_candidate_points(self, n_candidate_points, seed):
        sampler = qmc.LatinHypercube(d=2, seed=1)
        samples = sampler.random(n=n_candidate_points)
        l_bounds = [0, 0]
        u_bounds = [1, 1]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        candidate_points = jnp.array(scaled_samples)
        return candidate_points










class Poisson10D:
    """
    Defines the problem for a 2D Poisson equation.

    PDE:         Δu(x, y) = f(x, y)
    Domain:      (x, y) ∈ (0, 1)²
    BCs:         u(x, 0) = 0
                 ∂u/∂n(0, y) = ∂u/∂n(1, y) = ∂u/∂n(x, 1) = 0
    """
    # Problem parameters
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0

    def __init__(self):
        """Initializes the problem."""
        pass
    
    def exact_solution_point(self, x):
        u = 0

        for k in range(1, 6):
            idx1 = 2 * k - 2  # e.g., for k=1, idx1=0 (for x₁)
            idx2 = 2 * k - 1  # e.g., for k=1, idx2=1 (for x₂)
            term = x[idx1] * x[idx2]
            u += term

        return u
    
    def exact_solution_batch(self, X: np.ndarray):
        even_cols = X[:, 0::2]
        odd_cols = X[:, 1::2]
        u = np.sum(even_cols * odd_cols, axis=1)
        return u

    class FCN(eqx.Module):
        # 2. Declare all fields at the class level
        weights: list[jax.Array]
        biases: list[jax.Array]
        
        # Mark non-trainable attributes as static
        activation: Callable = eqx.field(static=True)
        """A simple fully connected neural network (MLP) using Equinox."""

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))

        def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
            """A standard forward pass. Hard constraints are omitted for simplicity."""
            
            nn_input = xs
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]
            
            sd = 0.025
            
            u = 0
            for k in range(1, 6):
                idx1 = 2 * k - 2
                idx2 = 2 * k - 1
                term = xs[idx1] * xs[idx2]
                u += term

            boundary_factor = 1.0
            for k in range(10):
                boundary_factor *= jnp.tanh(xs[k] / sd) * jnp.tanh((1 - xs[k]) / sd)
            
            return u * (1 - boundary_factor) + nn_output * boundary_factor

        def predict(self, points):
            _model_prediction = jax.vmap(self.__call__)
            return _model_prediction(points)


    @staticmethod
    def physics_residual(model: FCN, xs: jnp.ndarray) -> jnp.ndarray:
        def u_fn(coords):
            return model(coords).squeeze()

        laplacian_u = jnp.trace(jax.hessian(u_fn)(xs))

        source_term = 0
                           
        return laplacian_u - source_term

    
    def get_candidate_points(self, n_candidate_points, seed):
        sampler = qmc.LatinHypercube(d=10, seed=1)
        samples = sampler.random(n=n_candidate_points)
        l_bounds = [0] * 10
        u_bounds = [1] * 10
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        candidate_points = jnp.array(scaled_samples)
        return candidate_points














class Poisson3D:
    """
    Defines the problem for a 3D Poisson equation using a Physics-Informed Neural Network (PINN).

    PDE:        Δu(x, y, z) = -3π²sin(πx)sin(πy)sin(πz)
    Domain:     (x, y, z) ∈ (0, 1)³
    BCs:        u(x, y, z) = 0 on the boundary ∂(0, 1)³
    """
    # Problem domain parameters
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0

    def __init__(self):
        """Initializes the problem."""
        pass

    def exact_solution(self, nx: int = 32, ny: int = 32, nz: int = 32):
        vx = np.linspace(self.xmin, self.xmax, nx)
        vy = np.linspace(self.ymin, self.ymax, ny)
        vz = np.linspace(self.zmin, self.zmax, nz)
        X, Y, Z = np.meshgrid(vx, vy, vz, indexing='ij')

        u_grid = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

        return X, Y, Z, u_grid

    class FCN(eqx.Module):
        # 2. Declare all fields at the class level
        weights: list[jax.Array]
        biases: list[jax.Array]
        
        # Mark non-trainable attributes as static
        activation: Callable = eqx.field(static=True)
        """A simple fully connected neural network (MLP) using Equinox."""

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))

        def __call__(self, xyz: jnp.ndarray) -> jnp.ndarray:
            x, y, z = xyz[0], xyz[1], xyz[2]
            
            nn_input = xyz
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]

            # Enforce u=0 on all boundaries x,y,z = 0 or 1 using the requested tanh method
            sd = 0.05
            boundary_factor = (jax.nn.tanh(x / sd) * jax.nn.tanh((1 - x) / sd) *
                               jax.nn.tanh(y / sd) * jax.nn.tanh((1 - y) / sd) *
                               jax.nn.tanh(z / sd) * jax.nn.tanh((1 - z) / sd))

            return nn_output * boundary_factor

        def predict(self, X, Y, Z):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], Y.flatten()[:,None], Z.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape)

    # --- Derivative and Residual Calculation Utilities ---
    _u_hessian_fn = jax.hessian(lambda model, xyz: model(xyz), argnums=1)

    @staticmethod
    def physics_residual(model: FCN, xyz: jnp.ndarray) -> jnp.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]

        u_hessian = Poisson3D._u_hessian_fn(model, xyz)
        u_xx = u_hessian[0, 0]
        u_yy = u_hessian[1, 1]
        u_zz = u_hessian[2, 2]

        laplacian_u = u_xx + u_yy + u_zz
        f = -3 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)

        return laplacian_u - f

    def get_candidate_points(self, n_candidates, seed):
        """Generates a large, uniform pool of candidate points for resampling."""
        key = jax.random.PRNGKey(seed)
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.xmin, maxval=self.xmax)
        key, _ = jax.random.split(key)
        y_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.ymin, maxval=self.ymax)
        key, _ = jax.random.split(key)
        z_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.zmin, maxval=self.zmax)
        
        return jnp.hstack([x_candidates, y_candidates, z_candidates])







class Heat1_1D:
    def __init__(self):
        pass

    def exact_solution_at_point(self, x, t):
        kappa = 1.0 / 4.0
        initial_condition = jnp.sin(2 * jnp.pi * x)
        u = jnp.exp(-4.0 * jnp.pi**2 * kappa * t) * initial_condition
        return u

    class FCN(eqx.Module):
        weights: Tuple[jax.Array, ...]
        biases: Tuple[jax.Array, ...]
        activation: Callable = eqx.field(static=True)

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))

        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            x, t = xt[:1], xt[1]
            
            nn_input = xt
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]

            sd = 0.5
            
            time_factor = jax.nn.tanh(t / sd)
            boundary_factor = jnp.prod(jax.nn.tanh(x / sd) * jax.nn.tanh((1 - x) / sd))
            initial_condition_term = jnp.sum(jnp.sin(2 * jnp.pi * x))
            u = initial_condition_term * (1 - time_factor) + boundary_factor * time_factor * nn_output

            return u

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            flat_coords = [coord.flatten()[:, None] for coord in X]
            flat_t = T.flatten()[:, None]
            test_points = jnp.hstack(flat_coords + [flat_t])

            return _model_prediction(test_points).reshape(T.shape)

    @staticmethod
    def physics_residual(model, xt: jnp.ndarray) -> jnp.ndarray:
        kappa = 1.0 / 4.0
        
        def model_output_scalar(xt_scalar):
            return model(xt_scalar)

        grad_u_vector = jax.grad(model_output_scalar)(xt)
        u_t = grad_u_vector[1]
        
        def d2_u_dx2(i):
            return jax.grad(lambda xt: jax.grad(model_output_scalar)(xt)[i])(xt)[i]

        d2_u_dx2_vmap = jax.vmap(d2_u_dx2, in_axes=(0,))
        laplacian_u = jnp.sum(d2_u_dx2_vmap(jnp.arange(1)))
        
        residual = u_t - kappa * laplacian_u
        
        return residual

    def get_candidate_points(self, n_candidates, seed):
        key = jax.random.PRNGKey(seed)
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=0, maxval=1)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=0, maxval=1)
        
        return jnp.hstack([x_candidates, t_candidates])


class Heat2_1D:
    def __init__(self):
        pass

    def exact_solution_at_point(self, x, t):
        kappa = 1.0 / 4.0
        initial_condition = jnp.sum(jnp.sin(2 * jnp.pi * x))
        u = jnp.exp(-4.0 * jnp.pi**2 * kappa * t) * initial_condition
        return u

    class FCN(eqx.Module):
        weights: Tuple[jax.Array, ...]
        biases: Tuple[jax.Array, ...]
        activation: Callable = eqx.field(static=True)

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))

        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
            x, t = xt[:2], xt[2]
            
            nn_input = xt
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]

            sd = 0.1
            
            time_factor = jax.nn.tanh(t / sd)
            boundary_factor = jnp.prod(jax.nn.tanh(x / sd) * jax.nn.tanh((1 - x) / sd))
            initial_condition_term = jnp.sum(jnp.sin(2 * jnp.pi * x))
            u = initial_condition_term * (1 - time_factor) + boundary_factor * time_factor * nn_output

            return u

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            flat_coords = [coord.flatten()[:, None] for coord in X]
            flat_t = T.flatten()[:, None]
            test_points = jnp.hstack(flat_coords + [flat_t])

            return _model_prediction(test_points).reshape(T.shape)

    @staticmethod
    def physics_residual(model, xt: jnp.ndarray) -> jnp.ndarray:
        kappa = 1.0 / 4.0
        
        def model_output_scalar(xt_scalar):
            return model(xt_scalar)

        grad_u_vector = jax.grad(model_output_scalar)(xt)
        u_t = grad_u_vector[2]
        
        def d2_u_dx2(i):
            return jax.grad(lambda xt: jax.grad(model_output_scalar)(xt)[i])(xt)[i]

        d2_u_dx2_vmap = jax.vmap(d2_u_dx2, in_axes=(0,))
        laplacian_u = jnp.sum(d2_u_dx2_vmap(jnp.arange(2)))
        
        residual = u_t - kappa * laplacian_u
        
        return residual

    def get_candidate_points(self, n_candidates, seed):
        key = jax.random.PRNGKey(seed)
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 2), minval=0, maxval=1)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=0, maxval=1)
        
        return jnp.hstack([x_candidates, t_candidates])


class Heat10_1D:
    def __init__(self):
        """Initializes the problem."""
        pass

    def exact_solution_at_point(self, x, t):
        kappa = 1.0 / 4.0
        
        initial_condition_sum = jnp.prod(jnp.sin(2 * jnp.pi * x))
        u = jnp.exp(-40.0 * jnp.pi**2 * kappa * t) * initial_condition_sum
        return u

    class FCN(eqx.Module):
        # 2. Declare all fields at the class level
        weights: Tuple[jax.Array, ...]
        biases: Tuple[jax.Array, ...]
        
        # Mark non-trainable attributes as static
        activation: Callable = eqx.field(static=True)
        """A simple fully connected neural network (MLP) using Equinox."""

        def __init__(self, key, layer_sizes, activation=jax.nn.tanh):
            self.activation = activation
            self.weights = []
            self.biases = []
            initializer = jax.nn.initializers.glorot_uniform()

            for i in range(len(layer_sizes) - 1):
                key, w_key, b_key = jax.random.split(key, 3)
                
                in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
                self.weights.append(initializer(w_key, (out_dim, in_dim)))
                self.biases.append(jnp.zeros((out_dim,)))

        def __call__(self, xst: jnp.ndarray) -> jnp.ndarray:
            xs, t = xst[:10], xst[10]
            
            nn_input = xst
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                nn_input = self.activation(w @ nn_input + b)
            nn_input = self.weights[-1] @ nn_input + self.biases[-1]
            nn_output = nn_input[0]

            sd = 0.01
            boundary_factor = jnp.prod(jax.nn.tanh(xs / sd) * jax.nn.tanh((1 - xs) / sd))
            initial_condition_term = jnp.prod(jnp.sin(2 * jnp.pi * xs))
            u = initial_condition_term * (1 - jax.nn.tanh(t / 0.1)) + nn_output * jax.nn.tanh(t / 0.05)

            return u

        def predict(self, Xs, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            flat_coords = [coord.flatten()[:, None] for coord in Xs]
            flat_t = T.flatten()[:, None]
            test_points = jnp.hstack(flat_coords + [flat_t])

            return _model_prediction(test_points).reshape(T.shape)

    @staticmethod
    def physics_residual(model, xst: jnp.ndarray) -> jnp.ndarray:
        kappa = 1.0 / 4.0
        
        def model_output_scalar(xst_scalar):
            return model(xst_scalar)

        grad_u_vector = jax.grad(model_output_scalar)(xst)
        u_t = grad_u_vector[10]
        
        def d2_u_dx2(i):
            return jax.grad(lambda xst: jax.grad(model_output_scalar)(xst)[i])(xst)[i]

        d2_u_dx2_vmap = jax.vmap(d2_u_dx2, in_axes=(0,))
        laplacian_u = jnp.sum(d2_u_dx2_vmap(jnp.arange(10)))
        
        residual = u_t - kappa * laplacian_u
        
        return residual

    def get_candidate_points(self, n_candidates, seed):
        """Generates a large, uniform pool of candidate points for resampling."""
        key = jax.random.PRNGKey(seed)
        xs_candidates = jax.random.uniform(key, shape=(n_candidates, 10), minval=0, maxval=1)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=0, maxval=1)
        
        return jnp.hstack([xs_candidates, t_candidates])













class KovasznayFlow:
    # Problem domain parameters
    xmin: float = -0.5
    xmax: float = 1.0
    ymin: float = -0.5
    ymax: float = 1.5
    Re: float = 40.0

    def __init__(self):
        """Initializes the problem."""
        pass

    def exact_solution(self, nx: int = 256, ny: int = 256):
        nu = 1.0 / self.Re

        lambda_val = -0.963740544195767032178063367006170

        vx = np.linspace(self.xmin, self.xmax, nx)
        vy = np.linspace(self.ymin, self.ymax, ny)
        X, Y = np.meshgrid(vx, vy, indexing='ij')

        u = 1.0 - np.exp(lambda_val * X) * np.cos(2.0 * np.pi * Y)
        
        v = (lambda_val / (2.0 * np.pi)) * np.exp(lambda_val * X) * np.sin(2.0 * np.pi * Y)
        
        p = 0.5 * (1.0 - np.exp(2.0 * lambda_val * X))

        return X, Y, u, v, p

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh):
            """
            Initializes the FCN.
            
            Args:
                key: A JAX random key.
                layer_sizes: A list of integers defining the size of each layer.
                activation: The activation function to use between layers.
            """
            if layer_sizes is None:
                # Default network architecture: Input is (x,y,z), output is u
                layer_sizes = [2, 20, 20, 20, 20, 3]

            self.activation = activation
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))

        def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
            """
            Performs a forward pass and applies hard constraints for the boundary conditions.
            The Dirichlet boundary condition u=0 is enforced by multiplying the network output
            by a factor that is zero at the boundaries.
            """
            x, y = xy[0], xy[1]

            # 1. Perform the standard forward pass through the network
            # Assumes the network's final layer has 3 outputs for u, v, and p
            nn_input = xy
            for i, layer in enumerate(self.layers):
                nn_input = layer(nn_input)
                if i < len(self.layers) - 1:
                    nn_input = self.activation(nn_input)
            
            # Unpack the raw network outputs for u, v, and p
            nn_u, nn_v, nn_p = nn_input[0], nn_input[1], nn_input[2]

            sd = 0.05
            
            A = jax.nn.tanh((x - KovasznayFlow.xmin) / sd) * jax.nn.tanh((KovasznayFlow.xmax - x) / sd) * \
                jax.nn.tanh((y - KovasznayFlow.ymin) / sd) * jax.nn.tanh((KovasznayFlow.ymax - y) / sd)
            
            
            lambda_val = -0.963740544195767032178063367006170
            
            x_hat = (x - KovasznayFlow.xmin) / (KovasznayFlow.xmax - KovasznayFlow.xmin)
            y_hat = (y - KovasznayFlow.ymin) / (KovasznayFlow.ymax - KovasznayFlow.ymin)

            # Interpolation for u
            uxmin = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.cos(2 * jnp.pi * y)
            uxmax = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.cos(2 * jnp.pi * y)
            uymin = 1.0 - jnp.exp(lambda_val * x) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymin)
            uymax = 1.0 - jnp.exp(lambda_val * x) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymax)

            c1_u = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymin)
            c2_u = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymin)
            c3_u = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymax)
            c4_u = 1.0 - jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.cos(2 * jnp.pi * KovasznayFlow.ymax)

            B_u = ( (1 - x_hat) * uxmin + x_hat * uxmax +
                    (1 - y_hat) * uymin + y_hat * uymax -
                    ( (1 - x_hat) * (1 - y_hat) * c1_u +
                    x_hat * (1 - y_hat) * c2_u +
                    (1 - x_hat) * y_hat * c3_u +
                    x_hat * y_hat * c4_u ) )
            
            # Interpolation for v (Generalized Form)
            vxmin = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.sin(2 * jnp.pi * y)
            vxmax = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.sin(2 * jnp.pi * y)
            vymin = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * x) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymin)
            vymax = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * x) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymax)

            c1_v = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymin)
            c2_v = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymin)
            c3_v = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmin) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymax)
            c4_v = (lambda_val / (2 * jnp.pi)) * jnp.exp(lambda_val * KovasznayFlow.xmax) * jnp.sin(2 * jnp.pi * KovasznayFlow.ymax)

            B_v = ( (1 - x_hat) * vxmin + x_hat * vxmax +
                    (1 - y_hat) * vymin + y_hat * vymax -
                    ( (1 - x_hat) * (1 - y_hat) * c1_v +
                    x_hat * (1 - y_hat) * c2_v +
                    (1 - x_hat) * y_hat * c3_v +
                    x_hat * y_hat * c4_v ) )

            B_p = 0.5 * (1.0 - jnp.exp(2.0 * lambda_val * x))

            # final_solution = A(x,y) * NN(x,y) + B(x,y)
            final_u = A * nn_u + B_u
            final_v = A * nn_v + B_v
            final_p = A * nn_p + B_p

            # Stack the results into a single output tensor
            return jnp.stack([final_u, final_v, final_p])

        def predict(self, X, Y):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape + (3,))

    # --- Derivative and Residual Calculation Utilities ---
    _u_hessian_fn = jax.hessian(lambda model, xyz: model(xyz), argnums=1)

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0))
    def physics_residual(residual_fn: Callable, model: FCN, xy: jnp.ndarray) -> jnp.ndarray:
        u, v, p = model(xy)
        jac_uvp = jax.jacobian(model)(xy)
        du_dx, du_dy = jac_uvp[0, 0], jac_uvp[0, 1]
        dv_dx, dv_dy = jac_uvp[1, 0], jac_uvp[1, 1]
        dp_dx, dp_dy = jac_uvp[2, 0], jac_uvp[2, 1]
        
        # Laplacian for u
        u_fn = lambda xy_coords: model(xy_coords)[0] # Function that returns only u
        u_hessian = jax.jacobian(jax.grad(u_fn))(xy)
        laplacian_u = jnp.trace(u_hessian)

        # Laplacian for v
        v_fn = lambda xy_coords: model(xy_coords)[1] # Function that returns only v
        v_hessian = jax.jacobian(jax.grad(v_fn))(xy)
        laplacian_v = jnp.trace(v_hessian)

        residuals = residual_fn(
            u, v, p, du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy,
            laplacian_u, laplacian_v
        )
        return residuals

    @staticmethod
    def residual_fn(u, v, p, du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy,
                    laplacian_u, laplacian_v) -> jnp.ndarray:
        """
        Defines the residuals of the 2D steady, incompressible Navier-Stokes equations.

        Returns:
            A vector [Rx, Ry, Rc] where each element is the value of a residual.
        """
        nu = 1.0 / KovasznayFlow.Re
        R_x = (u * du_dx + v * du_dy) + dp_dx - (nu * laplacian_u)
        R_y = (u * dv_dx + v * dv_dy) + dp_dy - (nu * laplacian_v)
        R_c = du_dx + dv_dy

        return jnp.stack([R_x, R_y, R_c])

    def get_collocation_points(self, n: int, seed: int):
        """
        Generates uniformly distributed points within the 3D domain using Latin Hypercube Sampling.
        
        Args:
            n: The number of points to generate.
            seed: A seed for the random number generator.
            
        Returns:
            A JAX array of collocation points.
        """
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n=n)
        l_bounds = [self.xmin, self.ymin]
        u_bounds = [self.xmax, self.ymax]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        return jnp.array(scaled_samples)
    

class Lorenz:
    """
    Defines the problem for the Lorenz system of ODEs.

    The system is governed by a set of coupled ODEs:
    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz

    The solution is computed for t in [0, 20].
    """
    # Problem parameters
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    tmin: float = 0.0
    tmax: float = 20.0
    initial_condition: jnp.ndarray = jnp.array([1.0, 1.0, 1.0])

    def __init__(self):
        """Initializes the problem."""
        pass

    def get_reference_solution(self, nt: int = 1000):
        """
        Computes a high-fidelity numerical solution to serve as a reference.
        The Lorenz system does not have a simple analytical solution.
        """
        # Define the ODE system for the solver
        def lorenz_ode(t, xyz):
            x, y, z = xyz
            dxdt = self.sigma * (y - x)
            dydt = x * (self.rho - z) - y
            dzdt = x * y - self.beta * z
            return [dxdt, dydt, dzdt]

        # Time points for the solution
        t_span = [self.tmin, self.tmax]
        t_eval = np.linspace(self.tmin, self.tmax, nt)
        
        # Solve the ODE
        sol = solve_ivp(
            lorenz_ode, 
            t_span, 
            self.initial_condition, 
            t_eval=t_eval, 
            dense_output=True,
            method='RK45'
        )
        
        return sol.t, sol.y.T # Return time points and solution [x, y, z]

    def get_collocation_points(self, n: int, seed: jax.random.PRNGKey):
        """Generates training points in the time domain using Latin Hypercube Sampling."""
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        samples = sampler.random(n=n)
        scaled_samples = qmc.scale(samples, [self.tmin], [self.tmax])
        return jnp.array(scaled_samples).squeeze()

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)
        initial_condition: jnp.ndarray = eqx.field(static=True)

        def __init__(self, key: jax.random.PRNGKey, initial_condition: jnp.ndarray, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh):
            """
            Initializes the FCN.
            - `initial_condition` is required to enforce hard constraints.
            """
            if layer_sizes is None:
                # [time] -> [h1] -> [h2] -> [h3] -> [x, y, z]
                layer_sizes = [1, 64, 64, 64, 3]
            
            self.activation = activation
            self.initial_condition = initial_condition
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))

        def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
            """
            Forward pass of the PINN, with hard constraints for initial conditions.
            The network learns the deviation from the initial state.
            """
            nn_input = jnp.reshape(t, (1,))
            for i, layer in enumerate(self.layers):
                nn_input = layer(nn_input)
                if i < len(self.layers) - 1:
                    nn_input = self.activation(nn_input)
            
            # Hard constraint: u(t) = u(0) + t * NN(t)
            # This ensures that u(0) = u(0) automatically.
            # A smooth factor (1 - exp(-t)) can also be used instead of just 't'.
            u = self.initial_condition + t * nn_input
            
            return u

        def predict(self, T: jnp.ndarray):
            """Makes predictions over a batch of time points."""
            # vmap efficiently maps the model call over a batch of time points.
            return jax.vmap(self.__call__)(T)

    # Use jacfwd for efficient computation of the time derivative (Jacobian)
    _u_t_fn = jax.jacfwd(lambda model, t: model(t), argnums=1)

    @staticmethod
    def residual_fn(u: jnp.ndarray, u_t: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Calculates the physics residual for the Lorenz system."""
        x, y, z = u
        dx_dt, dy_dt, dz_dt = u_t
        
        sigma, rho, beta = params['sigma'], params['rho'], params['beta']
        
        res_x = dx_dt - sigma * (y - x)
        res_y = dy_dt - (x * (rho - z) - y)
        res_z = dz_dt - (x * y - beta * z)
        
        return jnp.array([res_x, res_y, res_z])

    @staticmethod
    @partial(jax.vmap, in_axes=(None, 0))
    def physics_residual(model: FCN, t: jnp.ndarray) -> jnp.ndarray:
        """Computes the physics residual at a single time point `t`."""
        # Get model output [x(t), y(t), z(t)]
        u = model(t)
        
        # Get time derivative [dx/dt, dy/dt, dz/dt]
        u_t = Lorenz._u_t_fn(model, t)
        
        # Pack parameters for the residual function
        params = {'sigma': Lorenz.sigma, 'rho': Lorenz.rho, 'beta': Lorenz.beta}
        
        return Lorenz.residual_fn(u, u_t, params)
    
    def get_candidate_points(self, n_candidates: int, seed: int):
        """Generates a large, uniform pool of candidate points for resampling."""
        key = jax.random.PRNGKey(seed)
        # Generate candidates uniformly across the time domain
        t_candidates = jax.random.uniform(key, shape=(n_candidates,), minval=self.tmin, maxval=self.tmax)
        return t_candidates


from scipy.fft import fft, ifft, fftfreq

    
class KuramotoSivashinsky:
    """
    Defines the problem for the Kuramoto-Sivashinsky (KS) equation.

    (Version using a trigonometric coordinate transformation to enforce periodicity)
    """
    # PDE and domain parameters
    alpha: float = 100.0 / 16.0
    beta: float = 100.0 / (16.0**2)
    gamma: float = 100.0 / (16.0**4)
    xmin: float = 0.0
    xmax: float = 2.0 * np.pi
    Lx: float = xmax - xmin

    def __init__(self):
        pass

    def exact_solution(self, nx: int = 64, nt: int = 64, tfinal=1.0):
        """
        Computes a high-fidelity numerical solution using a spectral method
        (Method of Lines with FFT for spatial derivatives).
        
        Args:
            nt (int): Number of time points to evaluate the solution at.
            nx (int): Number of spatial grid points.
            
        Returns:
            T (np.ndarray): Time meshgrid of shape (nt, nx).
            X (np.ndarray): Space meshgrid of shape (nt, nx).
            u_solution (np.ndarray): The solution array of shape (nt, nx).
        """
        # 1. Define the spatial grid and wave numbers for FFT
        x_grid = np.linspace(self.xmin, self.xmax, nx, endpoint=False)
        # Correctly define the integer wavenumbers for a domain of size 2*pi
        k = np.fft.fftfreq(nx) * nx
        
        # 2. Define the RHS of the ODE system (du/dt = F(u))
        def ks_rhs(t, u):
            # Compute derivatives in Fourier space
            u_hat = fft(u)
            ux_hat = 1j * k * u_hat
            uxx_hat = (1j * k)**2 * u_hat
            uxxxx_hat = (1j * k)**4 * u_hat
            
            # Transform derivatives back to real space
            ux = np.real(ifft(ux_hat))
            # uxx = np.real(ifft(uxx_hat))
            # uxxxx = np.real(ifft(uxxxx_hat))
            
            # Compute the non-linear term in real space
            u_ux = u * ux
            # Transform non-linear term to Fourier space
            u_ux_hat = fft(u_ux)

            # Compute the PDE residual (excluding u_t) in Fourier space
            # This is more stable than computing the full RHS in real space
            rhs_hat = -self.alpha * u_ux_hat - self.beta * uxx_hat - self.gamma * uxxxx_hat
            
            return np.real(ifft(rhs_hat))

        # 3. Set up the time domain and initial condition
        t_span = [0.0, tfinal]
        t_eval = np.linspace(0.0, tfinal, nt)
        u0 = jnp.cos(x_grid) * (1 + jnp.sin(x_grid))

        # 4. Solve the system of ODEs
        # The 'Radau' method is often recommended for stiff problems like this.
        sol = solve_ivp(
            ks_rhs,
            t_span,
            u0,
            t_eval=t_eval,
            method='Radau'
        )
        
        # 5. Create meshgrids for the output
        t_sol = sol.t
        x_sol = x_grid
        # The solver returns shape (nx, nt), so we transpose it to (nt, nx)
        u_solution = sol.y.T
        
        # Create meshgrids using the standard 'xy' indexing (default)
        X, T = np.meshgrid(x_sol, t_sol)

        return X, T, u_solution

    def get_candidate_points(self, n_candidates, seed, tmax):
        """Generates a large, uniform pool of candidate points for resampling."""
        key = jax.random.PRNGKey(seed)
        # Generate candidates uniformly across the domain
        x_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=self.xmin, maxval=self.xmax)
        key, _ = jax.random.split(key)
        t_candidates = jax.random.uniform(key, shape=(n_candidates, 1), minval=0, maxval=tmax)
        
        return jnp.hstack([x_candidates, t_candidates])

    class FCN(eqx.Module):
        """A simple fully connected neural network (MLP) using Equinox."""
        layers: list
        activation: Callable = eqx.field(static=True)
        tmin: float
        tmax: float

        def __init__(self, key: jax.random.PRNGKey, layer_sizes: list[int] = None, activation: Callable = jax.nn.tanh, tmax=1.0):
            if layer_sizes is None:
                layer_sizes = [3, 10, 10, 10, 10, 1]
            
            self.activation = activation
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=subkey))
            self.tmin = 0.0
            self.tmax = tmax

        def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:

            x, t = xt[0], xt[1]

            periodic_x = jnp.hstack([jnp.cos(x), jnp.sin(x)])
            
            features = jnp.hstack([t, periodic_x])
            for i, layer in enumerate(self.layers):
                features = layer(features)
                if i < len(self.layers) - 1:
                    features = self.activation(features)
            nn_output = features[0]
        
            u0 = jnp.cos(x) * (1 + jnp.sin(x))
            
            sd = 0.1
            time_factor = jax.nn.tanh(t / sd)

            u = (1 - time_factor) * u0 + time_factor * jax.nn.tanh((self.tmax - t) / sd) * nn_output
            
            return u

        def predict(self, X, T):
            @partial(jax.vmap)
            def _model_prediction(points):
                return self.__call__(points)

            test_points = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            return _model_prediction(test_points).reshape(X.shape)


    @staticmethod
    def _derivatives(model: FCN, xt: jnp.ndarray):
        """
        A robust, step-by-step function to compute all necessary derivatives.
        """
        # 1. Get value and first derivatives (gradient)
        u, grads = jax.value_and_grad(model)(xt)
        u_x, u_t = grads[0], grads[1]

        # # 2. Define a clean helper function for the x-derivative
        # u_x_fn = lambda xt_inner: jax.grad(model)(xt_inner)[0]
        
        # # 3. Differentiate the helper function to get u_xx
        # u_xx = jax.grad(u_x_fn)(xt)[0]

        # # 4. Repeat the process for the fourth derivative
        # u_xx_fn = lambda xt_inner: jax.grad(u_x_fn)(xt_inner)[0]
        # # u_xxx = jax.grad(u_xx_fn)(xt)[0]
        # u_xxx_fn = lambda xt_inner: jax.grad(u_xx_fn)(xt_inner)[0]
        # u_xxxx = jax.grad(u_xxx_fn)(xt)[0]

        return u, u_t, u_x#, u_xx, u_xxxx

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0))
    def physics_residual(residual_fn: Callable, model: FCN, xt: jnp.ndarray) -> jnp.ndarray:
        """Computes derivatives and passes them to the residual function."""
        # Unpack the xt vector at the top level
        # u, u_t, u_x, u_xx, u_xxxx = KuramotoSivashinsky._derivatives(model, xt)
        u, u_t, u_x = KuramotoSivashinsky._derivatives(model, xt)
        # return residual_fn(u, u_t, u_x, u_xx, u_xxxx)
        return residual_fn(u, u_t, u_x)
    
    @staticmethod
    # def residual_fn(u, u_t, u_x, u_xx, u_xxxx) -> jnp.ndarray:
    def residual_fn(u, u_t, u_x) -> jnp.ndarray:
        """Calculates the physics residual from the supplied derivatives."""
        return (u_t 
                + KuramotoSivashinsky.alpha * u * u_x)