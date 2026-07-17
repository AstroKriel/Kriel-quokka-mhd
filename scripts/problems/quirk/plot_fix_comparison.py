## { SCRIPT

##
## === DEPENDENCIES
##

import re
from pathlib import Path

import numpy
from jormi.ww_plots import manage_plots, style_plots

##
## === CONSTANTS
##

COMBO_OFF = "q26-b25-ppm_ep-no-carbuncle-fix"
COMBO_ON = "q26-b25-ppm_ep"
NCELLS = 128
MAX_TIME = 0.5
## a single representative post-shock time is enough once the IC is shown separately: we take the
## second-to-last of what would otherwise be 4 evenly-spaced samples, i.e. a time well after the
## front has formed but before it reaches the domain edge
NUM_TIME_SAMPLES = 4
NUM_BINS = 16  ## coarse enough to be legible, still finer than the shock's own transition width

ROOT_DIR = Path(__file__).parents[3]
DATASET_ROOT = ROOT_DIR / f"datasets/problems/quirk/ncells={NCELLS}"
FIGURE_PATH = ROOT_DIR / f"figures/problems/quirk/ncells={NCELLS}/carbuncle_fix_comparison.png"

##
## === HELPER FUNCTIONS
##


def load_snapshots(combo):
    dataset_dir = DATASET_ROOT / combo
    pressure_paths = sorted(
        dataset_dir.glob("extracted/pressure-slice=x_2-index=*.npz"),
        key=lambda path: int(re.search(r"index=(\d+)", path.stem).group(1)),
    )
    snapshots = []
    for pressure_path in pressure_paths:
        pressure_data = numpy.load(pressure_path)
        snapshots.append(
            {
                "step_time": float(pressure_data["step_time"]),
                "pressure": pressure_data["sarray_2d"],  ## shape (num_cells_x0, num_cells_x1)
            },
        )
    return snapshots


def pick_evenly_spaced(snapshots, num_snapshots):
    if len(snapshots) <= num_snapshots:
        return snapshots
    indices = sorted(set(numpy.linspace(0, len(snapshots) - 1, num_snapshots).round().astype(int)))
    return [snapshots[index] for index in indices]


def bin_profile(
    *,
    field_2d: numpy.ndarray,
    num_bins: int,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Bin a (x_0, x_1) field into `num_bins` windows along x_0.

    Pooling every cell in the x_0-window together with every cell along x_1 (rather than just
    averaging over x_1 at each x_0) lets the bin's standard deviation capture the real signal here:
    the broadband, cell-to-cell corrugation along x_0 is what actually distinguishes the carbuncle
    instability, and is an order of magnitude larger than the spread across x_1 at fixed x_0.
    """
    num_cells_x0 = field_2d.shape[0]
    bin_size = num_cells_x0 // num_bins
    x_0_centers, means, stds = [], [], []
    for bin_index in range(num_bins):
        window = field_2d[bin_index * bin_size:(bin_index + 1) * bin_size, :]
        x_0_centers.append((bin_index + 0.5) * bin_size / num_cells_x0)
        means.append(window.mean())
        stds.append(window.std())
    return numpy.asarray(x_0_centers), numpy.asarray(means), numpy.asarray(stds)


def plot_binned_errorbar(*, ax, snapshot, color, zorder):
    x_0_centers, means, stds = bin_profile(field_2d=snapshot["pressure"], num_bins=NUM_BINS)
    ax.errorbar(
        x_0_centers, means, yerr=stds,
        color=color, marker="o", markeredgecolor="black", markeredgewidth=0.7,
        linestyle="none", capsize=5, capthick=2.0, elinewidth=2.0, zorder=zorder,
    )


def plot_raw_profile(*, ax, snapshot, color, zorder):
    ## the IC is a clean step with no scatter to bin away, and binning would only smear its sharp
    ## front, so we plot it at full cell resolution instead
    num_cells = snapshot["pressure"].shape[0]
    x_0 = (numpy.arange(num_cells) + 0.5) / num_cells
    ax.plot(x_0, snapshot["pressure"].mean(axis=1), color=color, linewidth=1.5, linestyle="--", zorder=zorder)


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()

    snapshots_off = [snapshot for snapshot in load_snapshots(COMBO_OFF) if snapshot["step_time"] <= MAX_TIME]
    snapshots_on = [snapshot for snapshot in load_snapshots(COMBO_ON) if snapshot["step_time"] <= MAX_TIME]

    initial_snapshot = snapshots_off[0]  ## t=0 initial condition, identical for both variants
    representative_off = pick_evenly_spaced(snapshots_off, NUM_TIME_SAMPLES)[-2]
    representative_on = pick_evenly_spaced(snapshots_on, NUM_TIME_SAMPLES)[-2]

    fig, ax = manage_plots.create_figure(fig_scale=1.4)

    plot_raw_profile(ax=ax, snapshot=initial_snapshot, color="black", zorder=1)
    plot_binned_errorbar(ax=ax, snapshot=representative_off, color="red", zorder=2)
    plot_binned_errorbar(ax=ax, snapshot=representative_on, color="blue", zorder=3)

    ax.set_ylabel(r"$p$")
    ax.set_xlabel(r"$x_0$")

    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=300,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
