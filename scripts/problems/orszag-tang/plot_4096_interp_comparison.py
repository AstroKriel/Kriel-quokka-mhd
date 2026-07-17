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
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

## fixed compute + averaging (Q26 + Balsara2025b, our recommended scheme); only interpolation
## varies between the two 4096^2 runs available, so the split shows PPM vs PPM-EP directly
INTERPOLATIONS = ("ppm", "ppm_ep")
INTERPOLATION_LABELS = {"ppm": "PPM", "ppm_ep": "PPM-EP"}

## second of the three saved snapshots (t = 0.5, 0.85, 1.0); index differs slightly between the
## two runs since they take different timesteps to reach the same physical time
SNAPSHOT_INDEX_BY_INTERPOLATION = {"ppm": "0056825", "ppm_ep": "0056669"}

FIELD_LABEL = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
PALETTE_NAME = "cmr.eclipse"
PALETTE_RANGE = (0.0, 1.0)
VALUE_RANGE = (-2.5, -1.2)

## the computational domain is 1 x 1 in dimensionless units
AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))
NUM_CELLS = 4096

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / f"datasets/problems/orszag-tang/ncells={NUM_CELLS}"
FIGURE_PATH = ROOT_DIR / f"figures/problems/orszag-tang/ncells={NUM_CELLS}/interp_comparison-index=0056825.png"

##
## === HELPER FUNCTIONS
##


def load_slice(
    *,
    interpolation: str,
) -> numpy.ndarray:
    """Load the second saved snapshot (t = 0.85) for `q26-b25-<interpolation>`."""
    index = SNAPSHOT_INDEX_BY_INTERPOLATION[interpolation]
    slice_path = (
        DATASET_DIR / f"q26-b25-{interpolation}" / "diagnostics"
        / f"current_density_magnitude-slice=x_2-index={index}-amr_level=0.npz"
    )
    return numpy.load(slice_path)["sarray_2d"]


def compose_interp_split(
    *,
    upper_array: numpy.ndarray,
    lower_array: numpy.ndarray,
) -> numpy.ndarray:
    """Combine two (rot180-symmetric) fields into one array, split across the main diagonal.

    Neither half loses information relative to showing either field in full: each field already
    repeats itself (point-reflected) between the two halves, so one half per field is sufficient.
    """
    num_rows, num_cols = upper_array.shape
    upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return numpy.where(upper_mask, upper_array, lower_array)


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / NUM_CELLS
    slices = {
        interpolation: numpy.log10(cell_size * load_slice(interpolation=interpolation))
        for interpolation in INTERPOLATIONS
    }
    composite = compose_interp_split(
        upper_array=slices[INTERPOLATIONS[0]],
        lower_array=slices[INTERPOLATIONS[1]],
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(8, 8),
    )
    ax = axs[0, 0]
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=composite,
        data_format="xy",
        axis_bounds=AXIS_BOUNDS,
        cbar_bounds=VALUE_RANGE,
        palette_config=add_color.SequentialConfig(palette_name=PALETTE_NAME, palette_range=PALETTE_RANGE),
        add_cbar=True,
        cbar_label=FIELD_LABEL,
        cbar_side="top",
    )
    ax.plot(
        [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
        [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
        color="white",
        linewidth=0.6,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.05,
        y_pos=0.95,
        label=INTERPOLATION_LABELS[INTERPOLATIONS[0]],
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        text_size=18,
        text_color="white",
        box_color="black",
        box_alpha=0.6,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.95,
        y_pos=0.05,
        label=INTERPOLATION_LABELS[INTERPOLATIONS[1]],
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Bottom,
        text_size=18,
        text_color="white",
        box_color="black",
        box_alpha=0.6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
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
