"""
Defines PDE problems to solve

Each problem class must inherit from the Problem base class.
Each problem class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.nn
import jax.numpy as jnp
import numpy as np

from fbpinns.util.logger import logger
from fbpinns.traditional_solutions.analytical.burgers_solution import burgers_viscous_time_exact1
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_2D_pressure_second_order import seismicCPML2D


class Problem:
    """Base problem class to be inherited by different problem classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Applies optional constraining operator"""
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """Defines exact solution, if it exists"""
        raise NotImplementedError





class HarmonicOscillator1D(Problem):
    """Solves the time-dependent damped harmonic oscillator
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # boundary loss
        x_batch_boundary = jnp.array([0.]).reshape((1,1))
        u_boundary = jnp.array([1.]).reshape((1,1))
        ut_boundary = jnp.array([0.]).reshape((1,1))
        required_ujs_boundary = (
            (0,()),
            (0,(0,)),
        )

        return [[x_batch_phys, required_ujs_phys], [x_batch_boundary, u_boundary, ut_boundary, required_ujs_boundary]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # boundary loss
        _, uc, utc, u, ut = constraints[1]
        if len(uc):
            boundary = 1e6*jnp.mean((u-uc)**2) + 1e2*jnp.mean((ut-utc)**2)
        else:
            boundary = 0# if no boundary points are inside the active subdomains (i.e. u.shape[0]=0), jnp.mean returns nan

        return phys + boundary

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):

        d, w0 = all_params["static"]["problem"]["d"], all_params["static"]["problem"]["w0"]

        w = jnp.sqrt(w0**2-d**2)
        phi = jnp.arctan(-d/w)
        A = 1/(2*jnp.cos(phi))
        cos = jnp.cos(phi + w * x_batch)
        exp = jnp.exp(-d * x_batch)
        u = exp * 2 * A * cos

        return u


class HarmonicOscillator1DHardBC(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator using hard boundary conditions
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20, sd=0.1):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )
        return [[x_batch_phys, required_ujs_phys],]# only physics loss required in this case

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        u = 1 + (tanh(x/sd)**2) * u# applies hard BCs
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        return phys


class HarmonicOscillator1DInverse(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator inverse problem
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu_true":mu,
            "k":k,
            }
        trainable_params = {
            "mu":jnp.array(0.),# learn mu from constraints
            }

        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # data loss
        x_batch_data = jnp.linspace(0,1,13).astype(float).reshape((13,1))# use 13 observational data points
        u_data = HarmonicOscillator1DInverse.exact_solution(all_params, x_batch_data)
        required_ujs_data = (
            (0,()),
            )

        return [[x_batch_phys, required_ujs_phys], [x_batch_data, u_data, required_ujs_data]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["trainable"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # data loss
        _, uc, u = constraints[1]
        data = 1e6*jnp.mean((u-uc)**2)

        return phys + data




class BurgersEquation2D(Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        du       du        d^2 u
        -- + u * -- = nu * -----
        dt       dx        dx^2

        for -1.0 < x < +1.0, and 0 < t

        Boundary conditions:
        u(x,0) = - sin(pi*x)
        u(-1,t) = u(+1,t) = 0
    """

    @staticmethod
    def init_params(nu=0.01/jnp.pi, sd=0.1):

        static_params = {
            "dims":(1,2),
            "nu":nu,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(1,)),
            (0,(0,0)),
        )
        return [[x_batch_phys, required_ujs_phys],]


    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        x, t, tanh, sin, pi = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sin, jnp.pi
        u = tanh((x+1)/sd)*tanh((1-x)/sd)*tanh((t-0)/sd)*u - sin(pi*x)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        nu = all_params["static"]["problem"]["nu"]
        _, u, ux, ut, uxx = constraints[0]
        phys = ut + (u*ux) - (nu*uxx)
        return jnp.mean(phys**2)

    @staticmethod
    def residual_fn(all_params, constraints):
        nu = all_params["static"]["problem"]["nu"]
        _, u, ux, ut, uxx = constraints[0]
        phys = ut + (u*ux) - (nu*uxx)
        return phys

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        nu = all_params["static"]["problem"]["nu"]

        # use the burgers_solution code to compute analytical solution
        xmin,xmax = x_batch[:,0].min().item(), x_batch[:,0].max().item()
        tmin,tmax = x_batch[:,1].min().item(), x_batch[:,1].max().item()
        vx = np.linspace(xmin,xmax,batch_shape[0])
        vt = np.linspace(tmin,tmax,batch_shape[1])
        logger.info("Running burgers_viscous_time_exact1..")
        vu = burgers_viscous_time_exact1(nu, len(vx), vx, len(vt), vt)
        u = jnp.array(vu.flatten()).reshape((-1,1))
        return u


from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


class AllenCahnEquation2D(Problem):
    """
    Solves the time-dependent 1D Allen-Cahn equation and provides a
    high-fidelity numerical solution for comparison.

        du     d^2 u
        -- = e * ----- + u * (1 - u^2)
        dt     dx^2

        for -1.0 < x < +1.0, and 0 < t

    Initial Condition:
        u(x,0) = x^2 * cos(pi*x)
    
    Boundary Conditions:
        Periodic on x.
    """

    # Class-level cache for the high-fidelity solver's interpolator object.
    # This prevents re-running the slow numerical solver every time.
    _solver_interpolator = None

    @staticmethod
    def init_params(epsilon=1e-4, kappa=5.0, sd=0.1, N_solver=512, t_max_solver=1.0):
        """
        Initializes the static parameters for the problem.

        Args:
            epsilon (float): Parameter for the Allen-Cahn equation.
            kappa (float): Parameter for the reaction term in the Allen-Cahn equation.
            sd (float): Scaling factor for the constraining function.
            N_solver (int): Number of spatial points for the numerical ground truth solver.
            t_max_solver (float): Maximum time for the numerical ground truth solver.
        """
        static_params = {
            "dims": (1, 2),
            "epsilon": epsilon,
            "kappa": kappa,  # Added kappa parameter
            "sd": sd,
            "N_solver": N_solver,
            "t_max_solver": t_max_solver,
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Samples points in the domain to enforce the physics."""
        # Physics loss points
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),      # u
            (0, (1,)),    # du/dt
            (0, (0, 0)),  # d²u/dx²
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Applies hard constraints for initial conditions."""
        sd = all_params["static"]["problem"]["sd"]
        x, t = x_batch[:, 0:1], x_batch[:, 1:2]
        sin, tanh, pi = jnp.sin,jax.nn.tanh, jnp.pi
        
        # Enforce the initial condition u(x,0) = x^2 * sin(2*pi*x)
        initial_condition = x**2 * sin(2 * pi * x)
        # Use a smooth tanh function to fade from the initial condition at t=0
        # to the neural network solution for t>0.
        u = tanh((x+1)/sd) * tanh((1-x)/sd) * tanh(t / sd) * u + (1 - jax.nn.tanh(t / sd)) * initial_condition
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the physics-informed loss from the Allen-Cahn residual."""
        problem_params = all_params["static"]["problem"]
        epsilon = problem_params["epsilon"]
        kappa = problem_params["kappa"] # Use kappa from params
        _, u, ut, uxx = constraints[0]
        
        # Allen-Cahn equation residual: f = du/dt - epsilon*d^2u/dx^2 + kappa*(u^3 - u)
        f = ut - epsilon * uxx + kappa * (u**3 - u)
        
        # Return the mean squared error of the residual
        return jnp.mean(f**2)

    @staticmethod
    def _allen_cahn_rhs(t, u, epsilon, kappa, dx):
        """Helper function for the numerical ODE solver (Method of Lines)."""
        # Enforce periodic boundaries using np.roll
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        laplacian = (u_left - 2 * u + u_right) / dx**2
        
        # ### CORRECTED REACTION TERM ###
        # The equation is du/dt - ε*uxx + κ*(u³-u) = 0
        # So, du/dt = ε*uxx - κ*(u³-u)
        reaction = -kappa * (u**3 - u)
        return epsilon * laplacian + reaction

    @classmethod
    def exact_solution(cls, all_params, x_batch, batch_shape):
        """
        Provides a high-fidelity numerical solution to the Allen-Cahn equation.

        This method uses the Method of Lines and a high-quality ODE solver
        (scipy.integrate.solve_ivp) to compute a benchmark solution. The result
        is cached and interpolated to avoid re-computation.
        """
        # --- Caching Mechanism ---
        if cls._solver_interpolator is None:
            logger.info("No cached solution found. Running high-fidelity numerical solver...")
            
            problem_params = all_params["static"]["problem"]
            epsilon = problem_params["epsilon"]
            kappa = problem_params["kappa"]
            N = problem_params["N_solver"]
            t_max = problem_params["t_max_solver"]
            
            L = 2.0
            dx = L / N
            x_grid = np.linspace(-1.0, 1.0, N, endpoint=False)
            t_grid = np.linspace(0, t_max, int(t_max * 100) + 1)
            
            # Set the initial condition on the grid
            u0 = x_grid**2 * np.sin(2 * np.pi * x_grid)
            
            # Solve the system of ODEs
            sol = solve_ivp(
                fun=cls._allen_cahn_rhs,
                t_span=(0, t_max),
                y0=u0,
                t_eval=t_grid,
                method='BDF', # 'BDF' is good for stiff equations
                args=(epsilon, kappa, dx) # Pass kappa to the solver
            )
            
            u_solution_grid = sol.y.T
            
            logger.info("Solver finished. Caching the solution interpolator.")
            cls._solver_interpolator = RegularGridInterpolator(
                (t_grid, x_grid), u_solution_grid,
                bounds_error=False, fill_value=None
            )
        else:
            logger.info("Using cached high-fidelity solution.")
            
        # --- Interpolation ---
        x_coords = x_batch[:, 0]
        t_coords = x_batch[:, 1]
        points_to_interpolate = jnp.hstack([t_coords[:, None], x_coords[:, None]])
        u_exact = cls._solver_interpolator(points_to_interpolate)
        
        return jnp.array(u_exact).reshape((-1, 1))


class WaveEquationConstantVelocity3D(Problem):
    """Solves the time-dependent (2+1)D wave equation with constant velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveEquationConstantVelocity3D.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
            (0,(2,2)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        x, t = x_batch[:,0:2], x_batch[:,2:3]
        tanh, exp = jax.nn.tanh, jnp.exp

        # get starting wavefield
        p = jnp.expand_dims(source, axis=1)# (k, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)

        # form time-decaying anzatz
        t1 = source[:,2].min()/c0
        f = exp(-0.5*(1.5*t/t1)**2) * f
        t = tanh(2.5*t/t1)**2
        return f + t*u

    @staticmethod
    def loss_fn(all_params, constraints):
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt
        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        # use the seismicCPML2D FD code with very fine sampling to compute solution

        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        c_fn = params["c_fn"]

        (xmin, ymin, tmin), (xmax, ymax, tmax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))

        # get grid spacing
        deltax, deltay, deltat = (xmax-xmin)/(batch_shape[0]-1), (ymax-ymin)/(batch_shape[1]-1), (tmax-tmin)/(batch_shape[2]-1)

        # get f0, target deltas of FD simulation
        f0 = c0/source[:,2].min()# approximate frequency of wave
        DELTAX = DELTAY = 1/(f0*10)# target fine sampled deltas
        DELTAT = DELTAX / (4*np.sqrt(2)*c0)# target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax/DELTAX)), int(np.ceil(deltay/DELTAY)), int(np.ceil(deltat/DELTAT))# make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax/dx, deltay/dy, deltat/dt
        NX, NY, NSTEPS = batch_shape[0]*dx-(dx-1), batch_shape[1]*dy-(dy-1), batch_shape[2]*dt-(dt-1)

        # get starting wavefield
        xx,yy = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY), indexing="ij")# (NX, NY)
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        exp = np.exp
        p = np.expand_dims(source, axis=1)# (k, 1, 4)
        x = np.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        p0 = f.reshape((NX, NY))

        # get velocity model
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        c = np.array(c_fn(all_params, x))
        if c.shape[0]>1: c = c.reshape((NX, NY))
        else: c = c*np.ones_like(xx)

        # add padded CPML boundary
        NPOINTS_PML = 10
        p0 = np.pad(p0, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")
        c =   np.pad(c, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")

        # run simulation
        logger.info(f'Running seismicCPML2D {(NX, NY, NSTEPS)}..')
        wavefields, _ = seismicCPML2D(
                    NX+2*NPOINTS_PML,
                    NY+2*NPOINTS_PML,
                    NSTEPS,
                    DELTAX,
                    DELTAY,
                    DELTAT,
                    NPOINTS_PML,
                    c,
                    np.ones((NX+2*NPOINTS_PML,NY+2*NPOINTS_PML)),
                    (p0.copy(),p0.copy()),
                    f0,
                    np.float32,
                    output_wavefields=True,
                    gather_is=None)

        # get croped, decimated, flattened wavefields
        wavefields = wavefields[:,NPOINTS_PML:-NPOINTS_PML,NPOINTS_PML:-NPOINTS_PML]
        wavefields = wavefields[::dt, ::dx, ::dy]
        wavefields = np.moveaxis(wavefields, 0, -1)
        assert wavefields.shape == batch_shape
        u = wavefields.reshape((-1, 1))

        return u

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0 = all_params["static"]["problem"]["c0"]
        return jnp.array([[c0]], dtype=float)# (1,1) scalar value


class WaveEquationGaussianVelocity3D(WaveEquationConstantVelocity3D):
    """Solves the time-dependent (2+1)D wave equation with gaussian mixture velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]]), mixture=np.array([[0.5, 0.5, 1., 0.2]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveEquationGaussianVelocity3D.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            "mixture":jnp.array(mixture),# location, width and amplitude of gaussian pertubations in velocity model (l, 4)
            }
        return static_params, {}

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0, mixture = all_params["static"]["problem"]["c0"], all_params["static"]["problem"]["mixture"]
        x = x_batch[:,0:2]# (n, 2)
        exp = jnp.exp

        # get velocity model
        p = jnp.expand_dims(mixture, axis=1)# (l, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        c = c0 + f# (n, 1)
        return c




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from fbpinns.domains import RectangularDomainND

    np.random.seed(0)

    mixture=np.concatenate([
        np.random.uniform(-3, 3, (100,2)),# location
        0.4*np.ones((100,1)),# width
        0.3*np.random.uniform(-1, 1, (100,1)),# amplitude
        ], axis=1)

    source=np.array([# multiscale sources
        [0,0,0.1,1],
        [1,1,0.2,0.5],
        [-1.1,-0.5,0.4,0.25],
        ])

    # test wave equation
    for problem, kwargs in [(WaveEquationConstantVelocity3D, dict()),
                            (WaveEquationGaussianVelocity3D, dict(source=source, mixture=mixture))]:

        ps_ = problem.init_params(**kwargs)
        all_params = {"static":{"problem":ps_[0]}, "trainable":{"problem":ps_[1]}}

        batch_shape = (80,80,50)
        x_batch = RectangularDomainND._rectangle_samplerND(None, "grid", np.array([-3, -3, 0]), np.array([3, 3, 3]), batch_shape)

        plt.figure()
        c = np.array(problem.c_fn(all_params, x_batch))
        if c.shape[0]>1: c = c.reshape(batch_shape)
        else: c = c*np.ones(batch_shape)
        plt.imshow(c[:,:,0])
        plt.colorbar()
        plt.show()

        u = problem.exact_solution(all_params, x_batch, batch_shape).reshape(batch_shape)
        uc = np.zeros_like(x_batch)[:,0:1]
        uc = problem.constraining_fn(all_params, x_batch, uc).reshape(batch_shape)

        its = range(0,50,3)
        for u_ in [u, uc]:
            vmin, vmax = np.quantile(u, 0.05), np.quantile(u, 0.95)
            plt.figure(figsize=(2*len(its),5))
            for iplot,i in enumerate(its):
                plt.subplot(1,len(its),1+iplot)
                plt.imshow(u_[:,:,i], vmin=vmin, vmax=vmax)
            plt.show()
        plt.figure()
        plt.plot(u[40,40,:], label="u")
        plt.plot(uc[40,40,:], label="uc")
        t = np.linspace(0,1,50)
        plt.plot(np.tanh(2.5*t/(all_params["static"]["problem"]["source"][:,2].min()/all_params["static"]["problem"]["c0"]))**2)
        plt.legend()
        plt.show()



