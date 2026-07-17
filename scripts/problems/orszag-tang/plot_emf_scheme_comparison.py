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

## same grid as `plot_scheme_grid.py`, but comparing the two EMF averaging schemes via contour
## overlays (diagonal-split, each with a faint 'ghost' of itself over the other half) instead of
## an imshow + diagonal-split composite
EMF_RECONSTRUCTIONS = ("fs17", "b25", "q26")  ## grid columns (left -> right)
INTERPOLATIONS = ("plm", "ppm", "ppm_ep")  ## grid rows (top -> bottom)
EMF_AVERAGINGS = ("ld04", "b25")  ## (upper-left triangle, lower-right triangle)

RECONSTRUCTION_LABELS = {"fs17": "FS17", "b25": "B25", "q26": "Q26"}
INTERPOLATION_LABELS = {"plm": "PLM", "ppm": "PPM", "ppm_ep": "PPM-EP"}
AVERAGING_LABELS = {"ld04": "LD04", "b25": "B25"}

COLOR_UPPER = "blue"  ## LD04
COLOR_LOWER = "red"  ## B25
GHOST_ALPHA = 0.4

SLICE_GLOB = "current_density_magnitude-slice=x_2-index=*.npz"
## compare at the second of the three saved snapshots (t = 0.5, 0.85, 1.0), rather than the last
TARGET_TIME = 0.85
CONTOUR_LEVEL = -1.6

## the computational domain is 1 x 1 in dimensionless units
AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/orszag-tang/ncells=1024"
FIGURE_PATH = ROOT_DIR / "figures/problems/orszag-tang/ncells=1024/scheme_grid_contours.png"

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    scheme_dir: Path,
) -> Path:
    """Return the saved slice whose `step_time` is closest to `TARGET_TIME`.

    Different schemes take different numbers of (CFL-limited) timesteps to reach the same physical
    time, so the saved snapshot index that corresponds to `TARGET_TIME` differs slightly between
    combos; matching on the metadata directly (rather than assuming a shared index) is robust to that.
    """
    slice_paths = sorted((scheme_dir / "extracted").glob(SLICE_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {scheme_dir / 'extracted'}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - TARGET_TIME),
    )


def load_log_field(
    *,
    reconstruction: str,
    averaging: str,
    interpolation: str,
) -> numpy.ndarray:
    scheme_dir = DATASET_DIR / f"{reconstruction}-{averaging}-{interpolation}"
    slice_path = find_slice_near_time(scheme_dir=scheme_dir)
    array_2d = numpy.load(slice_path)["sarray_2d"]
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / array_2d.shape[0]
    return numpy.log10(cell_size * array_2d)


def add_averaging_comparison_contours(
    *,
    ax,
    field_upper: numpy.ndarray,
    field_lower: numpy.ndarray,
) -> None:
    """Overlay the two averaging schemes' contours, split across the main diagonal, each with a
    faint, full-domain 'ghost' of the same scheme drawn (semi-transparently) on top of the other half.
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
        levels=[CONTOUR_LEVEL], colors=COLOR_UPPER, linewidths=0.65, alpha=1.0, linestyles="solid", zorder=1,
    )
    ax.contour(
        grid_x, grid_y, field_lower_own.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_LOWER, linewidths=0.65, alpha=1.0, linestyles="solid", zorder=1,
    )
    ## faint 'ghost' contours in the opposite triangle (drawn on top)
    ax.contour(
        grid_x, grid_y, field_upper_ghost.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_UPPER, linewidths=0.65, alpha=GHOST_ALPHA, linestyles="solid", zorder=2,
    )
    ax.contour(
        grid_x, grid_y, field_lower_ghost.T,
        levels=[CONTOUR_LEVEL], colors=COLOR_LOWER, linewidths=0.65, alpha=GHOST_ALPHA, linestyles="solid", zorder=2,
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
    fig, axs = manage_plots.create_figure_grid(
        num_rows=len(INTERPOLATIONS),
        num_cols=len(EMF_RECONSTRUCTIONS),
        axis_shape=(4, 4),
        x_spacing=0.02,
        y_spacing=0.02,
        share_x=True,
        share_y=True,
    )
    for row_index, interpolation in enumerate(INTERPOLATIONS):
        for col_index, reconstruction in enumerate(EMF_RECONSTRUCTIONS):
            ax = axs[row_index, col_index]
            field_upper = load_log_field(
                reconstruction=reconstruction,
                averaging=EMF_AVERAGINGS[0],
                interpolation=interpolation,
            )
            field_lower = load_log_field(
                reconstruction=reconstruction,
                averaging=EMF_AVERAGINGS[1],
                interpolation=interpolation,
            )
            add_averaging_comparison_contours(ax=ax, field_upper=field_upper, field_lower=field_lower)
            ax.set_facecolor("white")
            ax.set_xlim(AXIS_BOUNDS[0])
            ax.set_ylim(AXIS_BOUNDS[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.05, 0.95, AVERAGING_LABELS[EMF_AVERAGINGS[0]],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=22, color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
            )
            ax.text(
                0.95, 0.05, AVERAGING_LABELS[EMF_AVERAGINGS[1]],
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=22, color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
            )
            if row_index == 0:
                ax.set_title(RECONSTRUCTION_LABELS[reconstruction])
            if col_index == 0:
                ax.set_ylabel(INTERPOLATION_LABELS[interpolation])
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
