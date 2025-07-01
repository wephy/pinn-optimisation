"""
Imports and calls various plotting functions depending on the dimensionality of the FBPINN / PINN problem

This module is used by trainers.py
"""

from fbpinns import plot_trainer_1D, plot_trainer_2D, plot_trainer_3D

_plotters = {
    "FBPINN":{1: plot_trainer_1D.plot_1D_FBPINN,
              2: plot_trainer_2D.plot_2D_FBPINN,
              3: plot_trainer_3D.plot_3D_FBPINN,
        },
    "PINN":  {1: plot_trainer_1D.plot_1D_PINN,
              2: plot_trainer_2D.plot_2D_PINN,
              3: plot_trainer_3D.plot_3D_PINN,
        },
    }

def plot(trainer, dims, *args, **kwargs):
    "Plots FBPINN and PINN results"

    nx = dims[1]
    if trainer in _plotters and nx in _plotters[trainer]:
        # Pass along both positional and keyword arguments
        return _plotters[trainer][nx](*args, **kwargs)
    else:
        return ()
