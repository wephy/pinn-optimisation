"""
Defines the Constants object which defines and stores a problem setup and all of its hyperparameters
for both FBPINNs and PINNs

This constants object should be passed to the appropriate trainer class defined in trainers.py

This module is used by trainers.py
"""

import socket

import numpy as np
import optax

from fbpinns import domains, problems, decompositions, networks, schedulers
from fbpinns.constants_base import ConstantsBase


# helper functions

def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


# main constants class

class Constants(ConstantsBase):

    def __init__(self, **kwargs):
        "Defines global constants for model"

        # Define run
        self.run = "test"

        # Define domain
        self.domain = domains.RectangularDomainND
        self.domain_init_kwargs = dict(
            xmin=np.array([0.]),
            xmax=np.array([1.])
            )

        # Define problem
        self.problem = problems.HarmonicOscillator1D
        #self.problem = problems.HarmonicOscillator1DInverse
        self.problem_init_kwargs = dict(
            d=2,
            w0=20,
            )

        # Define domain decomposition
        subdomain_xs = [np.linspace(0,1,5)]
        subdomain_ws = get_subdomain_ws(subdomain_xs, 2.99)
        self.decomposition = decompositions.RectangularDecompositionND
        self.decomposition_init_kwargs = dict(
            subdomain_xs=subdomain_xs,
            subdomain_ws=subdomain_ws,
            unnorm=(0., 1.),
            )

        # Define neural network
        self.network = networks.FCN
        self.network_init_kwargs = dict(
            layer_sizes=[1, 32, 1],
            )
        
        # Define training schedule
        self.training_schedule = [
            [
                (schedulers.AllActiveSchedulerND, dict()),
                (optax.adam, 10000, dict(learning_rate=1e-3))
            ],
            # Example of a second stage with a different scheduler and optimizer
            # [
            #     (schedulers.RandomSubdomainScheduler, dict(steps_per_subdomain=100)),
            #     (lbfgs, 10000, dict()) # Note: L-BFGS requires a different update function
            # ]
        ]

        # Define optimisation parameters
        self.ns = ((60,),)# batch_shape for each training constraint
        self.n_test = (200,)# batch_shape for test data
        self.sampler = "grid"# JAX uniform sampler is recommended for on-the-fly generation
        self.seed = 0

        # Define summary output parameters
        self.summary_freq    = 100# outputs train stats to command line
        self.test_freq       = 1000# outputs test stats to plot / file / command line
        self.model_save_freq = 1e10
        self.show_figures = True# whether to show figures
        self.save_figures = False# whether to save figures
        self.clear_output = True# whether to clear ipython output periodically

        # other constants
        self.hostname = socket.gethostname().lower()
        
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase
        
        self.n_steps = sum(stage[1][1] for stage in self.training_schedule)


if __name__ == "__main__":

    c = Constants(seed=2)
    print(c)

    c.get_outdirs()
    c.save_constants_file()