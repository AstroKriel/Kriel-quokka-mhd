## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

from matplotlib import colorbar as mpl_colorbar
from matplotlib import ticker as mpl_ticker
from numpy import typing as numpy_typing
from scipy import ndimage as scipy_ndimage

## personal
from jormi.ww_arrays import mask_2d_arrays
from jormi.ww_io import manage_io
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.snapshots import find_snapshots

## local
from local_helpers import paper_style, plot_slices

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class DensitySlice:
    step_time: float
    log10_density: numpy_typing.NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/blast-wave"
FIGURE_PATH = ROOT_DIR / "figures/problems/blast-wave/resolution-comparison.png"
NUM_CELLS_UPPER = 128
NUM_CELLS_LOWER = 1024

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-0.5, 0.5), (-0.5, 0.5))
MAJOR_TICK_STEP = 0.25
MINOR_TICK_STEP = 0.05
LABELED_TICK_VALUES = (-0.25, 0.25)
CONTOUR_LEVELS = (-0.0075, 0.0075)
CONTOUR_COLORS = ("blue", "red")

##
## === HELPER FUNCTIONS
##


def load_density_slice(
    *,
    num_cells: int,
) -> DensitySlice:
    """Load the density slice nearest `target_time` for one resolution."""
    extracted_dir = DATASET_DIR / f"num_cells={num_cells}" / "q26-b25-ppm_ep" / "extracted"
    slice_path = find_snapshots.find_npz_near_time(
        extracted_dir=extracted_dir,
        glob_pattern="density-slice=x_2-index=*.npz",
        target_time=0.05,
    )
    with numpy.load(slice_path) as data:
        return DensitySlice(
            step_time=float(data["step_time"]),
            log10_density=numpy.log10(data["sarray_2d"]),
        )


def combine_arrays_split_diagonally(
    *,
    upper_array: numpy_typing.NDArray[numpy.floating],
    lower_array: numpy_typing.NDArray[numpy.floating],
) -> numpy_typing.NDArray[numpy.floating]:
    upper_array = plot_slices.upsample_slice(
        array_2d=upper_array,
        target_num_cells=lower_array.shape[0],
    )
    num_rows, num_cols = upper_array.shape
    upper_mask = mask_2d_arrays.DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return numpy.where(upper_mask, upper_array, lower_array)


def compute_smoothed_slice(
    *,
    array_2d: numpy_typing.NDArray[numpy.floating],
) -> numpy_typing.NDArray[numpy.floating]:
    """
    Smooth by a fixed physical length (not a fixed cell count), so background noise is suppressed
    the same way at every resolution, rather than the coarser grid looking artificially cleaner.
    """
    domain_width = AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]
    cell_size = domain_width / array_2d.shape[0]
    smoothing_length = 0.1 * domain_width / NUM_CELLS_UPPER
    smoothing_sigma = smoothing_length / cell_size
    return scipy_ndimage.gaussian_filter(array_2d, sigma=smoothing_sigma)


def overlay_contours(
    *,
    panel: manage_figure.Panel,
    slice_2d: numpy_typing.NDArray[numpy.floating],
) -> None:
    num_rows, num_cols = slice_2d.shape
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1], num_cols),
        numpy.linspace(AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1], num_rows),
    )
    panel.contour(
        grid_x,
        grid_y,
        slice_2d.T,
        levels=list(CONTOUR_LEVELS),
        colors=list(CONTOUR_COLORS),
        linestyles="solid",
        linewidths=0.5,
        alpha=0.35,
        zorder=1,
    )


def mark_contour_levels_on_cbar(
    *,
    cbar: mpl_colorbar.Colorbar,
) -> None:
    for level, color in zip(CONTOUR_LEVELS, CONTOUR_COLORS, strict=True):
        cbar.ax.axvline(
            x=level,
            color=color,
            linewidth=0.5,
            zorder=10,
        )


def compute_zero_centred_palette_range(
    *,
    value_range: tuple[float, float],
) -> tuple[float, float]:
    """
    Return a `palette_range` that puts `value=0` at the colormap's own centre (0.5) via a single
    linear map (no piecewise/stretched norm): the palette end on the smaller-magnitude side of
    zero is pulled in from its natural extreme (0.0 or 1.0); the larger-magnitude side keeps its
    full extreme.
    """
    value_lo, value_hi = value_range
    if not (value_lo < 0.0 < value_hi):
        raise ValueError(f"`value_range` must straddle zero, got {value_range}.")
    zero_fraction = -value_lo / (value_hi - value_lo)
    if zero_fraction >= 0.5:
        return (0.0, 0.5 / zero_fraction)
    return ((0.5 - zero_fraction) / (1.0 - zero_fraction), 1.0)


def configure_domain_ticks(
    *,
    panel: manage_figure.Panel,
) -> None:
    figure_params = style_figure.get_figure_params()
    theme_params = figure_params.theme_params
    for axis in (panel.xaxis, panel.yaxis):
        axis.set_major_locator(mpl_ticker.MultipleLocator(MAJOR_TICK_STEP))
        axis.set_minor_locator(mpl_ticker.MultipleLocator(MINOR_TICK_STEP))
        axis.set_major_formatter(
            plot_slices.make_domain_tick_formatter(labeled_tick_values=LABELED_TICK_VALUES),
        )
    panel.tick_params(
        which="both",
        color=theme_params.foreground_color,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    figure_params = style_figure.get_figure_params()
    frame_params = figure_params.frame_params
    panel_gaps = figure_params.figure_layout.panel_gaps
    text_size_params = figure_params.text_size_params
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    upper_slice = load_density_slice(num_cells=NUM_CELLS_UPPER)
    lower_slice = load_density_slice(num_cells=NUM_CELLS_LOWER)
    palette_name = "blue-white-red"
    cbar_bounds = (-0.8, 0.55)
    palette_config = add_color.SequentialConfig(
        palette_name=palette_name,
        palette_range=compute_zero_centred_palette_range(value_range=cbar_bounds),
    )
    figure, panel = manage_figure.create_figure(
        panel_aspect_ratio=1.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    composite = combine_arrays_split_diagonally(
        upper_array=upper_slice.log10_density,
        lower_array=lower_slice.log10_density,
    )
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=composite,
        data_format="xy",
        axis_ranges=AXIS_BOUNDS,
        colorbar_range=cbar_bounds,
        palette_config=palette_config,
        add_colorbar=False,
    )
    smoothed_composite = combine_arrays_split_diagonally(
        upper_array=compute_smoothed_slice(array_2d=upper_slice.log10_density),
        lower_array=compute_smoothed_slice(array_2d=lower_slice.log10_density),
    )
    overlay_contours(
        panel=panel,
        slice_2d=smoothed_composite,
    )
    panel.plot(
        [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
        [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
        color="white",
        linewidth=frame_params.line_width_pt,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.05,
        y_pos_fraction=0.95,
        label=rf"${NUM_CELLS_UPPER}^3$",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        text_size_pt=text_size_params.axis_label_size_pt,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.95,
        y_pos_fraction=0.05,
        label=rf"${NUM_CELLS_LOWER}^3$",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Bottom,
        text_size_pt=text_size_params.axis_label_size_pt,
    )
    configure_domain_ticks(panel=panel)
    panel.set_xlabel(r"$x_0$")
    panel.set_ylabel(r"$x_1$")
    palette = add_color.make_palette(
        config=palette_config,
        value_range=cbar_bounds,
    )
    cbar = add_color.add_colorbar(
        panels=panel,
        palette=palette,
        label=r"$\log_{10}(\rho / \rho_\mathrm{bg})$",
        colorbar_side="top",
        colorbar_gap_pt=panel_gaps.row_pt / 2.0,
        label_gap_pt=frame_params.axis_label_gap_pt * 2.0,
    )
    mark_contour_levels_on_cbar(cbar=cbar)
    manage_figure.save_figure(
        figure=figure,
        figure_path=FIGURE_PATH,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
