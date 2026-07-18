## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_arrays.mask_2d_arrays import DiagonalMasks2D
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots, style_plots

##
## === CONFIGURATION
##

## same comparison as `plot_4096_interp_comparison.py` (PPM vs PPM-EP interpolation, for our
## recommended Q26 + B25 EMF combination, at the second saved snapshot, t = 0.85), but using
## contour overlays (diagonal-split, each with a faint 'ghost' of itself over the other half)
## instead of an imshow + diagonal-split composite
INTERPOLATIONS = ("ppm", "ppm_ep")  ## (upper-left triangle, lower-right triangle)
INTERPOLATION_LABELS = {"ppm": "PPM", "ppm_ep": "PPM-EP"}
## second of the three saved snapshots (t = 0.5, 0.85, 1.0); index differs slightly between the
## two runs since they take different timesteps to reach the same physical time
SNAPSHOT_INDEX_BY_INTERPOLATION = {"ppm": "0056825", "ppm_ep": "0056669"}

COLOR_UPPER = "blue"  ## PPM
COLOR_LOWER = "red"  ## PPM-EP
GHOST_ALPHA = 0.4

CONTOUR_LEVEL = -1.6

## the computational domain is 1 x 1 in dimensionless units
AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))
NUM_CELLS = 4096

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / f"datasets/problems/orszag-tang/ncells={NUM_CELLS}"
FIGURE_PATH = ROOT_DIR / f"figures/problems/orszag-tang/ncells={NUM_CELLS}/interp_comparison_contours-index=0056825.png"

##
## === HELPER FUNCTIONS
##


def load_log_field(
    *,
    interpolation: str,
) -> numpy.ndarray:
    """Load the second saved snapshot (t = 0.85) for `q26-b25-<interpolation>`."""
    index = SNAPSHOT_INDEX_BY_INTERPOLATION[interpolation]
    slice_path = (
        DATASET_DIR / f"q26-b25-{interpolation}" / "extracted"
        / f"current_density_magnitude-slice=x_2-index={index}-amr_level=0.npz"
    )
    array_2d = numpy.load(slice_path)["sarray_2d"]
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / NUM_CELLS
    return numpy.log10(cell_size * array_2d)


def add_interp_comparison_contours(
    *,
    ax,
    field_upper: numpy.ndarray,
    field_lower: numpy.ndarray,
) -> None:
    """Overlay the two interpolation schemes' contours, split across the main diagonal, each with
    a faint, full-domain 'ghost' of the same scheme drawn (semi-transparently) on top of the other half.
    """
    num_rows, num_cols = field_upper.shape
    upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(num_rows=num_rows, num_cols=num_cols)
    lower_mask = DiagonalMasks2D.get_mask_below_main_diagonal(num_rows=num_rows, num_cols=num_cols)
    field_upper_own = numpy.where(upper_mask, field_upper, numpy.nan)
    field_upper_ghost = numpy.where(lower_mask, field_upper, numpy.nan)
    field_lower_own = numpy.where(lower_mask, field_lower, numpy.nan)
    field_lower_ghost = numpy.where(upper_mask, field_lower, numpy.nan)
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1], num_cols),
        numpy.linspace(AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1], num_rows),
    )
    ## primary, fully-opaque contours in each scheme's own triangle (drawn first, in the back)
    ax.contour(
        grid_x, grid_y, field_upper_own.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_UPPER, linewidths=1, alpha=1.0, linestyles="solid", zorder=1,
    )
    ax.contour(
        grid_x, grid_y, field_lower_own.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_LOWER, linewidths=1, alpha=1.0, linestyles="solid", zorder=1,
    )
    ## faint 'ghost' contours in the opposite triangle (drawn on top)
    ax.contour(
        grid_x, grid_y, field_upper_ghost.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_UPPER, linewidths=1, alpha=GHOST_ALPHA, linestyles="solid", zorder=2,
    )
    ax.contour(
        grid_x, grid_y, field_lower_ghost.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_LOWER, linewidths=1, alpha=GHOST_ALPHA, linestyles="solid", zorder=2,
    )
    ax.plot(
        [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
        [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
        color="black",
        linewidth=0.6,
        zorder=3,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    field_upper = load_log_field(interpolation=INTERPOLATIONS[0])
    field_lower = load_log_field(interpolation=INTERPOLATIONS[1])
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(8, 8),
    )
    ax = axs[0, 0]
    add_interp_comparison_contours(ax=ax, field_upper=field_upper, field_lower=field_lower)
    ax.set_facecolor("white")
    ax.set_xlim(AXIS_BOUNDS[0])
    ax.set_ylim(AXIS_BOUNDS[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.05, 0.95, INTERPOLATION_LABELS[INTERPOLATIONS[0]],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=24, color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )
    ax.text(
        0.95, 0.05, INTERPOLATION_LABELS[INTERPOLATIONS[1]],
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=24, color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=400,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
