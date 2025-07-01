import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import LogNorm

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

from matplotlib.colors import LogNorm

# Helper function to plot the history data
def _plot_history(ax, history_data, title, y_label):
    """Plots a history list (step, value, optimizer_name) on a given axis."""
    if not history_data:
        ax.text(0.5, 0.5, "No history data", ha='center', va='center')
        ax.set_title(title)
        return

    # Group data by optimizer
    optimizer_groups = defaultdict(lambda: ([], []))
    for step, value, opt_name in history_data:
        optimizer_groups[opt_name][0].append(step)
        optimizer_groups[opt_name][1].append(value)

    # Plot each optimizer's data as a separate line
    for opt_name, (steps, values) in optimizer_groups.items():
        # Ensure values are positive for log scale
        safe_values = [v for v in values if v > 0]
        safe_steps = [s for s, v in zip(steps, values) if v > 0]
        if safe_values:
            ax.plot(safe_steps, safe_values, marker='.', linestyle='-', label=opt_name)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")


def _plot_test_im(u_test, xlim, ulim, n_test, it=None, norm=None):
    u_test = u_test.reshape(n_test)
    if it is not None:
        u_test = u_test[:,:,it]# for 3D

    if norm is not None:
        plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
            origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
            cmap="viridis", norm=norm)
    else:
        plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
            origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
            cmap="viridis", vmin=ulim[0], vmax=ulim[1])
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test, train_loss_history, test_error_history):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    f = plt.figure(figsize=(12, 18))

    # Row 1: Domain and Raw Output
    ax1 = plt.subplot(4,2,1)
    ax1.set_title(f"[{i}] Collocation Points")
    ax1.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    ax1.set_xlim(xlim[0][0], xlim[1][0])
    ax1.set_ylim(xlim[0][1], xlim[1][1])
    ax1.set_aspect("equal")

    ax2 = plt.subplot(4,2,2)
    ax2.set_title(f"[{i}] Raw Sub-Network Output")
    if us_raw_test is not None and us_raw_test.size > 0:
        ax2.hist(us_raw_test.flatten(), bins=100, label=f"min/max:\n{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    ax2.legend(loc=1)
    ax2.set_xlim(-5,5)

    # Row 2: Solution Plots
    ax3 = plt.subplot(4,2,3)
    ax3.set_title(f"[{i}] FBPINN Solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    ax4 = plt.subplot(4,2,4)
    ax4.set_title(f"[{i}] Ground Truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)
    
    # Row 3: Difference and Error
    ax5 = plt.subplot(4,2,5)
    ax5.set_title(f"[{i}] Absolute Difference")
    difference = np.abs(u_exact - u_test)
    vmax = difference.max() if difference.max() > 1e-4 else 1e-4
    _plot_test_im(difference, xlim0, ulim, n_test, norm=LogNorm(vmin=1e-5, vmax=vmax))
    
    ax6 = plt.subplot(4,2,6)
    error_l2 = np.linalg.norm(u_exact - u_test) / np.linalg.norm(u_exact)
    ax6.text(0.5, 0.5, f'L2 Relative Error:\n{error_l2:.4e}',
                 fontsize=15, ha='center', va='center')
    ax6.axis('off')
    
    # Row 4: History Plots
    ax7 = plt.subplot(4,2,7)
    _plot_history(ax7, train_loss_history, "Training Loss History", "Loss (log scale)")

    ax8 = plt.subplot(4,2,8)
    _plot_history(ax8, test_error_history, "Test Error History", "L2 Relative Error (log scale)")

    plt.tight_layout()
    return (("test",f),)

    # xlim, ulim = _plot_setup(x_batch_test, u_exact)
    # xlim0 = x_batch.min(0), x_batch.max(0)

    # f = plt.figure(figsize=(14,5))

    # plt.subplot(2,3,1)
    # plt.title(f"[{i}] Full solution")
    # _plot_test_im(u_test, xlim0, ulim, n_test)

    # plt.subplot(2,3,2)
    # plt.title(f"[{i}] Ground truth")
    # _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plt.subplot(2,3,3)
    # plt.title(f"[{i}] Difference")
    # difference = np.abs(u_exact - u_test)
    # _plot_test_im(difference, xlim0, ulim, n_test, norm=LogNorm(vmin=1e-4))

    # error_l2 = np.linalg.norm(u_exact - u_test) / np.linalg.norm(u_exact)

    # ax_text = plt.subplot(2, 1, 2)
    # ax_text.text(0.5, 0.5, f'L2 Relative Error: {error_l2:.4e}',
    #             fontsize=15,
    #             ha='center', # Horizontal alignment
    #             va='center') # Vertical alignment
    # ax_text.axis('off')

    # plt.tight_layout()

    # return (("test",f),)


@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test, train_loss_history=None, test_error_history=None):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    f = plt.figure(figsize=(12, 10))

    # --- Top Row: Solution Plots ---
    plt.subplot(3,2,1)
    plt.title(f"[{i}] PINN Solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,2)
    plt.title(f"[{i}] Ground Truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)
    
    plt.subplot(3,2,3)
    plt.title(f"[{i}] Absolute Difference")
    difference = np.abs(u_exact - u_test)
    vmax = difference.max() if difference.max() > 1e-4 else 1e-4
    _plot_test_im(difference, xlim0, ulim, n_test, norm=LogNorm(vmin=1e-5, vmax=vmax))

    ax_text = plt.subplot(3, 2, 4)
    error_l2 = np.linalg.norm(u_exact - u_test) / np.linalg.norm(u_exact)
    ax_text.text(0.5, 0.5, f'L2 Relative Error:\n{error_l2:.4e}',
                 fontsize=15, ha='center', va='center')
    ax_text.axis('off')

    # --- Bottom Row: History Plots ---
    ax5 = plt.subplot(3,2,5)
    _plot_history(ax5, train_loss_history, "Training Loss History", "Loss (log scale)")

    ax6 = plt.subplot(3,2,6)
    _plot_history(ax6, test_error_history, "Test Error History", "L2 Relative Error (log scale)")

    plt.tight_layout()
    return (("test",f),)
