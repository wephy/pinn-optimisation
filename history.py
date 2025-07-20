from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

class History:
    """
    A class to dynamically record and plot optimization metrics.

    This class does not require predefined metric names. It intelligently
    adapts its plotting style based on the dimensionality of the data recorded
    for each metric key.
    """
    def __init__(self):
        """Initializes the History object with an empty history dictionary."""
        self.history: Dict[str, List[Any]] = {}

    def record(self, metrics: Dict[str, Any]):
        """
        Records a dictionary of metrics from a single iteration.

        Args:
            metrics: A dictionary where keys are metric names (str) and
                     values are the metric values (scalar or array-like).
        """
        for key, value in metrics.items():
            # If the key is new, initialize its history with an empty list.
            if key not in self.history:
                self.history[key] = []
            # Append the new value, ensuring it's a NumPy array for consistency.
            self.history[key].append(np.asarray(value))

    def plot(self):
        """
        Generates and displays plots for all recorded metrics.

        - For metrics with scalar values per iteration, it plots a line graph.
        - For metrics with vector/array values per iteration, it plots the
          distribution (min, max, median) over iterations.
        """
        if not self.history:
            print("No history to plot.")
            return

        num_plots = len(self.history)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4.5), squeeze=False, sharex=True)
        axes = axes.flatten()
        fig.suptitle('Optimisation History', fontsize=16, y=0.98)

        for i, (key, data) in enumerate(self.history.items()):
            ax = axes[i]
            # Use the key to create a clean title, e.g., 'search_direction_norm' -> 'Search Direction Norm'
            title = key.replace('_', ' ').title()
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # --- Smart Plotting Logic ---
            # Check the shape of the first data entry to decide plot type.
            # np.size() correctly handles scalars (size 1) and arrays.
            if np.size(data[0]) == 1:
                self._plot_scalar(ax, data)
            else:
                self._plot_distribution(ax, data)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _plot_scalar(self, ax: plt.Axes, data: List[np.ndarray]):
        """Helper method to create a line plot for scalar data."""
        ax.plot(data, 'b-')
        ax.set_ylabel('Value')
        # Use a log scale for data spanning several orders of magnitude, but not for binary flags.
        flat_data = np.array(data).flatten()
        if np.min(flat_data) > 0 and np.max(flat_data) / np.min(flat_data) > 100:
             ax.set_yscale('log')

    def _plot_distribution(self, ax: plt.Axes, data: List[np.ndarray]):
        """Helper method to create a distribution plot for vector data."""
        # Stack list of 1D arrays into a 2D array for vectorized operations.
        values_matrix = np.array(data)
        
        s_max = np.max(values_matrix, axis=1)
        s_min = np.min(values_matrix, axis=1)
        s_median = np.median(values_matrix, axis=1)
        
        iterations = np.arange(len(data))
        
        ax.plot(iterations, s_max, 'r--', label='Max')
        ax.plot(iterations, s_median, 'k-', label='Median')
        ax.plot(iterations, s_min, 'b--', label='Min')
        ax.fill_between(iterations, s_min, s_max, color='gray', alpha=0.3)
        
        ax.set_ylabel('Value Distribution')
        ax.set_yscale('log') # Distributions like singular values are often best viewed on a log scale.
        ax.legend()
