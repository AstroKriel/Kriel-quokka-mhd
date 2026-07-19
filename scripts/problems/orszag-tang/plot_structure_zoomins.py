## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass, field
from pathlib import Path

## third-party
import numpy
from matplotlib import gridspec as mpl_gridspec
from matplotlib import patches as mpl_patches
from matplotlib import pyplot as mpl_plot

from matplotlib.figure import Figure as mpl_Figure
from numpy.typing import NDArray

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, manage_plots, plot_data, style_plots
from jormi.ww_validation import validate_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class BoundedSlice:
    sarray_2d: NDArray[numpy.floating]
    axis_bounds: plot_data.AxisBounds


@dataclass
class FigureGrid:
    num_rows: int
    num_cols: int
    fig_size: tuple[float, float]
    fig: mpl_Figure = field(init=False)
    grid_spec: mpl_gridspec.GridSpec = field(init=False)
    claimed: NDArray[numpy.bool_] = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        self.fig = mpl_plot.figure(figsize=self.fig_size)
        self.grid_spec = mpl_gridspec.GridSpec(
            nrows=self.num_rows,
            ncols=self.num_cols,
            figure=self.fig,
            left=0.0,
            right=1.0,
            bottom=0.0,
            top=1.0,
            wspace=0.01,
            hspace=0.01,
        )
        self.claimed = numpy.zeros((self.num_rows, self.num_cols), dtype=numpy.bool_)

    def add_field_axis(
        self,
        *,
        row_slice: slice,
        col_slice: slice,
        bounded_slice: BoundedSlice,
        add_cbar: bool,
    ) -> manage_plots.PlotAxis:
        """
        Reserve `row_slice`, `col_slice` for a new axis, and plot `bounded_slice` onto it.

        Raises `ValueError` if any cell in that zoom_region has already been claimed by a prior call.
        """
        if self.claimed[row_slice, col_slice].any():
            raise ValueError(f"grid zoom_region rows={row_slice}, cols={col_slice} is already claimed.")
        self.claimed[row_slice, col_slice] = True
        ax = self.fig.add_subplot(self.grid_spec[row_slice, col_slice])
        plot_sarray_2d(
            ax=ax,
            bounded_slice=bounded_slice,
            add_cbar=add_cbar,
        )
        return ax


@dataclass(frozen=True)
class ZoomRegion:
    x_lo: float
    y_lo: float
    x_hi: float
    y_hi: float

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_ordered_pair(
            (self.x_lo, self.x_hi),
            param_name="<x_lo, x_hi>",
            strict_ordering=True,
        )
        validate_types.ensure_ordered_pair(
            (self.y_lo, self.y_hi),
            param_name="<y_lo, y_hi>",
            strict_ordering=True,
        )


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATA_PATH: Path = (
    ROOT_DIR / "datasets/problems/orszag-tang/ncells=8192/q26-b25-ppm/extracted" /
    "current_density_magnitude-slice=x_2-index=0116573-amr_level=0.npz"
)
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/orszag-tang/ncells=8192/q26-b25-ppm/structure-zoomins.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisBounds = ((-0.5, 0.5), (-0.5, 0.5))

## annotations
ZOOM_REGIONS: tuple[ZoomRegion, ...] = (
    ZoomRegion(
        x_lo=0.0525,
        y_lo=-0.080875,
        x_hi=0.135,
        y_hi=0.001625,
    ),
    ZoomRegion(
        x_lo=0.2725,
        y_lo=-0.2485,
        x_hi=0.4225,
        y_hi=-0.0985,
    ),
)

##
## === HELPER FUNCTIONS
##


def crop_to_region(
    *,
    bounded_slice: BoundedSlice,
    zoom_region: ZoomRegion,
) -> BoundedSlice:
    num_cells_x, num_cells_y = bounded_slice.sarray_2d.shape
    domain_x_lo, domain_x_hi = bounded_slice.axis_bounds[0]
    domain_y_lo, domain_y_hi = bounded_slice.axis_bounds[1]
    index_x_lo = round((zoom_region.x_lo - domain_x_lo) / (domain_x_hi - domain_x_lo) * num_cells_x)
    index_x_hi = round((zoom_region.x_hi - domain_x_lo) / (domain_x_hi - domain_x_lo) * num_cells_x)
    index_y_lo = round((zoom_region.y_lo - domain_y_lo) / (domain_y_hi - domain_y_lo) * num_cells_y)
    index_y_hi = round((zoom_region.y_hi - domain_y_lo) / (domain_y_hi - domain_y_lo) * num_cells_y)
    cropped_sarray_2d = bounded_slice.sarray_2d[index_x_lo:index_x_hi, index_y_lo:index_y_hi]
    return BoundedSlice(
        sarray_2d=cropped_sarray_2d,
        axis_bounds=((zoom_region.x_lo, zoom_region.x_hi), (zoom_region.y_lo, zoom_region.y_hi)),
    )


def overlay_zoomin_box(
    *,
    ax: manage_plots.PlotAxis,
    bounds: plot_data.AxisBounds,
) -> None:
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    ax.add_patch(
        mpl_patches.Rectangle(
            (x_lo, y_lo),
            x_hi - x_lo,
            y_hi - y_lo,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
        ),
    )


def plot_sarray_2d(
    *,
    ax: manage_plots.PlotAxis,
    bounded_slice: BoundedSlice,
    add_cbar: bool,
) -> None:
    field_config = add_color.SequentialConfig(
        palette_name="cmr.wildfire_r",
        palette_range=(0.2, 1.0),
    )
    field_label = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=bounded_slice.sarray_2d,
        data_format="xy",
        axis_bounds=bounded_slice.axis_bounds,
        cbar_bounds=(-3.0, -1.25),
        palette_config=field_config,
        add_cbar=add_cbar,
        cbar_label=field_label if add_cbar else None,
        cbar_side="top",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_structures(
    *,
    bounded_slice: BoundedSlice,
) -> mpl_Figure:
    num_zoomin_rows = len(ZOOM_REGIONS)
    num_zoomin_cols = 1
    num_main_rows = num_zoomin_rows
    num_main_cols = num_main_rows
    num_rows = num_main_rows
    num_cols = num_main_cols + num_zoomin_cols
    cell_size_inches = 3.5
    fig_grid = FigureGrid(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_size=(
            cell_size_inches * num_cols,
            cell_size_inches * num_rows,
        ),
    )
    main_ax = fig_grid.add_field_axis(
        row_slice=slice(0, num_main_rows),
        col_slice=slice(0, num_main_cols),
        bounded_slice=bounded_slice,
        add_cbar=True,
    )
    for region_index, zoom_region in enumerate(ZOOM_REGIONS):
        grid_row_start = region_index
        grid_row_end = region_index + 1
        cropped_field = crop_to_region(
            bounded_slice=bounded_slice,
            zoom_region=zoom_region,
        )
        fig_grid.add_field_axis(
            row_slice=slice(grid_row_start, grid_row_end),
            col_slice=slice(num_main_cols, num_cols),
            bounded_slice=cropped_field,
            add_cbar=False,
        )
        overlay_zoomin_box(
            ax=main_ax,
            bounds=cropped_field.axis_bounds,
        )
    return fig_grid.fig


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme(theme=style_plots.Theme.LIGHT)
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    sarray_2d = numpy.load(DATA_PATH)["sarray_2d"]
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / sarray_2d.shape[0]
    log10_scaled_sarray_2d = compute_array_stats.compute_safe_log10(cell_size * numpy.abs(sarray_2d))
    bounded_slice = BoundedSlice(
        sarray_2d=log10_scaled_sarray_2d,
        axis_bounds=AXIS_BOUNDS,
    )
    fig = plot_structures(bounded_slice=bounded_slice)
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
