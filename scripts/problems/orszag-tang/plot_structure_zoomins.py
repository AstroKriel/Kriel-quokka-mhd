## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass, field
from pathlib import Path

## third-party
import numpy

from matplotlib import colors as mpl_colors
from matplotlib import figure as mpl_figure
from matplotlib import gridspec as mpl_gridspec
from matplotlib import patches as mpl_patches
from matplotlib import pyplot as mpl_plot
from matplotlib import ticker as mpl_ticker
from numpy import typing as numpy_typing

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_types
from ww_quokka_sims.sim_io import find_snapshots

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class BoundedSlice:
    sarray_2d: numpy_typing.NDArray[numpy.floating]
    axis_ranges: plot_data.AxisRanges


@dataclass
class FigureGrid:
    num_rows: int
    num_cols: int
    figure_size: tuple[float, float]
    figure_margins: manage_figure.FigureMargins
    panel_gaps: style_figure.PanelGaps
    figure: mpl_figure.Figure = field(init=False)
    grid_spec: mpl_gridspec.GridSpec = field(init=False)
    claimed: numpy_typing.NDArray[numpy.bool_] = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        self.figure = mpl_plot.figure(figsize=self.figure_size)
        figure_width = self.figure_size[0] * style_figure.PT_PER_INCH
        figure_height = self.figure_size[1] * style_figure.PT_PER_INCH
        panel_width = (
            figure_width - self.figure_margins.left_pt - self.figure_margins.right_pt -
            (self.num_cols - 1) * self.panel_gaps.col_pt
        ) / self.num_cols
        panel_height = (
            figure_height - self.figure_margins.bottom_pt - self.figure_margins.top_pt -
            (self.num_rows - 1) * self.panel_gaps.row_pt
        ) / self.num_rows
        self.grid_spec = mpl_gridspec.GridSpec(
            nrows=self.num_rows,
            ncols=self.num_cols,
            figure=self.figure,
            left=self.figure_margins.left_pt / figure_width,
            right=1.0 - self.figure_margins.right_pt / figure_width,
            bottom=self.figure_margins.bottom_pt / figure_height,
            top=1.0 - self.figure_margins.top_pt / figure_height,
            wspace=self.panel_gaps.col_pt / panel_width,
            hspace=self.panel_gaps.row_pt / panel_height,
        )
        self.claimed = numpy.zeros((self.num_rows, self.num_cols), dtype=numpy.bool_)

    def plot(
        self,
        *,
        row_slice: slice,
        col_slice: slice,
        bounded_slice: BoundedSlice,
        add_colorbar: bool,
    ) -> manage_figure.Panel:
        """
        Reserve `row_slice`, `col_slice` for a new axis, and plot `bounded_slice` onto it.

        Raises `ValueError` if any cell in that zoom_region has already been claimed by a prior call.
        """
        if self.claimed[row_slice, col_slice].any():
            raise ValueError(f"grid zoom_region rows={row_slice}, cols={col_slice} is already claimed.")
        self.claimed[row_slice, col_slice] = True
        panel = self.figure.add_subplot(self.grid_spec[row_slice, col_slice])
        plot_sarray_2d(
            panel=panel,
            bounded_slice=bounded_slice,
            add_colorbar=add_colorbar,
        )
        return panel


@dataclass(frozen=True)
class ZoomRegion:
    """
    A square zoom-in region: `width` gives both the x- and y-extent, from `x_lo`/`y_lo`.

    `y_label_step` thins the y tick labels to those landing on a multiple of it, leaving the
    ticks themselves alone; every tick is labelled when it is not given.
    """

    x_lo: float
    y_lo: float
    width: float
    y_label_step: float | None = None

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_float(
            param=self.width,
            param_name="width",
            require_positive=True,
            allow_zero=False,
        )

    @property
    def x_hi(
        self,
    ) -> float:
        return self.x_lo + self.width

    @property
    def y_hi(
        self,
    ) -> float:
        return self.y_lo + self.width


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
EXTRACTED_DIR: Path = ROOT_DIR / "datasets/problems/orszag-tang/num_cells=8192/q26-b25-ppm/extracted"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/orszag-tang/ncells=8192/q26-b25-ppm/structure-zoomins.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-0.5, 0.5), (-0.5, 0.5))
MAIN_LABELED_TICK_VALUES_X = (-0.5, -0.25, 0, 0.25)
MAIN_LABELED_TICK_VALUES_Y = (-0.5, -0.25, 0, 0.25, 0.5)
MAIN_MAJOR_TICK_STEP = 0.25
MAIN_MINOR_TICK_STEP = 0.05
PANEL_GAP = 2.5
FIGURE_ASPECT_RATIO = 171.0 / 125.0
FIGURE_MARGINS = manage_figure.FigureMargins(
    left_pt=42.0,
    right_pt=36.0,
    bottom_pt=28.0,
    top_pt=54.0,
)

## annotations
ZOOM_REGIONS: tuple[ZoomRegion, ...] = (
    ZoomRegion(
        x_lo=-0.1275,
        y_lo=0.005,
        width=0.07,
    ),
    ZoomRegion(
        x_lo=0.275,
        y_lo=-0.25,
        width=0.15,
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
    domain_x_lo, domain_x_hi = bounded_slice.axis_ranges[0]
    domain_y_lo, domain_y_hi = bounded_slice.axis_ranges[1]
    index_x_lo = round((zoom_region.x_lo - domain_x_lo) / (domain_x_hi - domain_x_lo) * num_cells_x)
    index_x_hi = round((zoom_region.x_hi - domain_x_lo) / (domain_x_hi - domain_x_lo) * num_cells_x)
    index_y_lo = round((zoom_region.y_lo - domain_y_lo) / (domain_y_hi - domain_y_lo) * num_cells_y)
    index_y_hi = round((zoom_region.y_hi - domain_y_lo) / (domain_y_hi - domain_y_lo) * num_cells_y)
    cropped_sarray_2d = bounded_slice.sarray_2d[index_x_lo:index_x_hi, index_y_lo:index_y_hi]
    return BoundedSlice(
        sarray_2d=cropped_sarray_2d,
        axis_ranges=((zoom_region.x_lo, zoom_region.x_hi), (zoom_region.y_lo, zoom_region.y_hi)),
    )


def overlay_zoomin_box(
    *,
    panel: manage_figure.Panel,
    bounds: plot_data.AxisRanges,
) -> None:
    figure_params = style_figure.get_figure_params()
    frame_params = figure_params.frame_params
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    panel.add_patch(
        mpl_patches.Rectangle(
            (x_lo, y_lo),
            x_hi - x_lo,
            y_hi - y_lo,
            facecolor=mpl_colors.to_rgba("white", alpha=0.15),
            edgecolor="white",
            linewidth=frame_params.line_width_pt,
            zorder=3,
        ),
    )


def connect_zoomin_to_panel(
    *,
    main_panel: manage_figure.Panel,
    zoom_panel: manage_figure.Panel,
    bounds: plot_data.AxisRanges,
) -> None:
    """
    Join each left corner of a marked region to the matching corner of its zoom panel.

    The lines belong to the main panel rather than the figure, so that the box can be drawn
    over them; a figure-level artist would always land on top of everything an axis holds.
    """
    figure_params = style_figure.get_figure_params()
    frame_params = figure_params.frame_params
    (x_lo, _), (y_lo, y_hi) = bounds
    for region_corner, panel_corner in (
        ((x_lo, y_hi), (0.0, 1.0)),
        ((x_lo, y_lo), (0.0, 0.0)),
    ):
        main_panel.add_artist(
            mpl_patches.ConnectionPatch(
                xyA=region_corner,
                coordsA=main_panel.transData,
                xyB=panel_corner,
                coordsB=zoom_panel.transAxes,
                color="white",
                linewidth=frame_params.line_width_pt,
                clip_on=False,
                zorder=2,
            ),
        )


def plot_sarray_2d(
    *,
    panel: manage_figure.Panel,
    bounded_slice: BoundedSlice,
    add_colorbar: bool,
) -> None:
    palette_config = add_color.SequentialConfig(
        palette_name="cmr.wildfire_r",
        palette_range=(0.2, 1.0),
    )
    field_label = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
    cbar_bounds = (-3.0, -1.25)
    clipped_sarray_2d = numpy.clip(bounded_slice.sarray_2d, cbar_bounds[0], cbar_bounds[1])
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=clipped_sarray_2d,
        data_format="xy",
        axis_ranges=bounded_slice.axis_ranges,
        colorbar_range=cbar_bounds,
        palette_config=palette_config,
        add_colorbar=False,
    )
    if add_colorbar:
        figure_params = style_figure.get_figure_params()
        frame_params = figure_params.frame_params
        palette = add_color.make_palette(
            config=palette_config,
            value_range=cbar_bounds,
        )
        add_color.add_colorbar(
            panels=panel,
            palette=palette,
            label=field_label,
            colorbar_side="top",
            colorbar_gap_pt=PANEL_GAP,
            label_gap_pt=frame_params.axis_label_gap_pt * 2.0,
        )


def format_main_ticks_x(
    tick_value: float,
    _tick_position: int,
) -> str:
    """
    Label only `MAIN_LABELED_TICK_VALUES`; every other major tick is drawn unlabeled.

    Labels are math mode, so their minus signs match the ones Matplotlib formats itself.
    """
    is_labeled = any(numpy.isclose(tick_value, labeled_value) for labeled_value in MAIN_LABELED_TICK_VALUES_X)
    return f"${tick_value:.2f}$" if is_labeled else ""


def format_main_ticks_y(
    tick_value: float,
    _tick_position: int,
) -> str:
    """Label only `MAIN_LABELED_TICK_VALUES`; every other major tick is drawn unlabeled."""
    is_labeled = any(numpy.isclose(tick_value, labeled_value) for labeled_value in MAIN_LABELED_TICK_VALUES_Y)
    return f"${tick_value:.2f}$" if is_labeled else ""


def configure_main_ticks(
    *,
    panel: manage_figure.Panel,
) -> None:
    for axis in (panel.xaxis, panel.yaxis):
        axis.set_major_locator(mpl_ticker.MultipleLocator(MAIN_MAJOR_TICK_STEP))
        axis.set_minor_locator(mpl_ticker.MultipleLocator(MAIN_MINOR_TICK_STEP))
    panel.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(format_main_ticks_x))
    panel.yaxis.set_major_formatter(mpl_ticker.FuncFormatter(format_main_ticks_y))
    panel.tick_params(
        which="both",
        color="white",
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )


def make_step_tick_formatter(
    *,
    label_step: float,
) -> mpl_ticker.FuncFormatter:
    """Label only ticks landing on a multiple of `label_step`; draw the rest unlabeled."""

    def format_tick(
        tick_value: float,
        _tick_position: int,
    ) -> str:
        nearest_multiple = round(tick_value / label_step) * label_step
        is_labeled = numpy.isclose(tick_value, nearest_multiple)
        return f"${tick_value:.2f}$" if is_labeled else ""

    return mpl_ticker.FuncFormatter(format_tick)


def configure_zoomin_ticks(
    *,
    panel: manage_figure.Panel,
    label_bottom: bool,
    label_top: bool,
    y_label_step: float | None,
) -> None:
    """Add automatic domain ticks (each zoom region's extent is too small for a fixed step)."""
    for axis in (panel.xaxis, panel.yaxis):
        axis.set_minor_locator(mpl_ticker.AutoMinorLocator())
    if y_label_step is not None:
        panel.yaxis.set_major_formatter(make_step_tick_formatter(label_step=y_label_step))
    panel.tick_params(
        which="both",
        color="white",
        labelbottom=label_bottom,
        labeltop=label_top,
        labelleft=False,
        labelright=True,
    )


def add_time_label(
    *,
    panel: manage_figure.Panel,
    step_time: float,
) -> None:
    figure_params = style_figure.get_figure_params()
    text_size_params = figure_params.text_size_params
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.5,
        y_pos_fraction=0.875,
        label=rf"$t = {step_time:.2f}$",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Top,
        text_color="white",
        text_size_pt=text_size_params.axis_label_size_pt,
    )


def plot_structures(
    *,
    bounded_slice: BoundedSlice,
    step_time: float,
) -> mpl_figure.Figure:
    num_zoomin_rows = len(ZOOM_REGIONS)
    num_zoomin_cols = 1
    num_main_rows = num_zoomin_rows
    num_main_cols = num_main_rows
    num_rows = num_main_rows
    num_cols = num_main_cols + num_zoomin_cols
    figure_width_inches = style_figure.FULL_PAGE_FIGURE_LAYOUT.figure_width.width_cm / style_figure.CM_PER_INCH
    panel_gaps = style_figure.PanelGaps(
        row_pt=PANEL_GAP,
        col_pt=PANEL_GAP,
    )
    figure_grid = FigureGrid(
        num_rows=num_rows,
        num_cols=num_cols,
        figure_size=(
            figure_width_inches,
            figure_width_inches / FIGURE_ASPECT_RATIO,
        ),
        figure_margins=FIGURE_MARGINS,
        panel_gaps=panel_gaps,
    )
    main_ax = figure_grid.plot(
        row_slice=slice(0, num_main_rows),
        col_slice=slice(0, num_main_cols),
        bounded_slice=bounded_slice,
        add_colorbar=True,
    )
    main_ax.set_xlabel(r"$x_0$")
    main_ax.set_ylabel(r"$x_1$")
    configure_main_ticks(panel=main_ax)
    add_time_label(
        panel=main_ax,
        step_time=step_time,
    )
    for region_index, zoom_region in enumerate(ZOOM_REGIONS):
        grid_row_start = region_index
        grid_row_end = region_index + 1
        cropped_field = crop_to_region(
            bounded_slice=bounded_slice,
            zoom_region=zoom_region,
        )
        zoom_ax = figure_grid.plot(
            row_slice=slice(grid_row_start, grid_row_end),
            col_slice=slice(num_main_cols, num_cols),
            bounded_slice=cropped_field,
            add_colorbar=False,
        )
        configure_zoomin_ticks(
            panel=zoom_ax,
            label_bottom=(region_index == len(ZOOM_REGIONS) - 1),
            label_top=(region_index == 0),
            y_label_step=zoom_region.y_label_step,
        )
        overlay_zoomin_box(
            panel=main_ax,
            bounds=cropped_field.axis_ranges,
        )
        connect_zoomin_to_panel(
            main_panel=main_ax,
            zoom_panel=zoom_ax,
            bounds=cropped_field.axis_ranges,
        )
    return figure_grid.figure


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params(
        figure_params=style_figure.FigureParams(theme=style_figure.Theme.LIGHT),
    )
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    data_path = find_snapshots.find_npz_near_time(
        extracted_dir=EXTRACTED_DIR,
        glob_pattern="current_density_magnitude-slice=x_2-index=*-amr_level=0.npz",
        target_time=0.85,
    )
    with numpy.load(data_path) as data:
        sarray_2d = data["sarray_2d"]
        step_time = float(data["step_time"])
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / sarray_2d.shape[0]
    log10_scaled_sarray_2d = compute_array_stats.compute_safe_log10(cell_size * numpy.abs(sarray_2d))
    bounded_slice = BoundedSlice(
        sarray_2d=log10_scaled_sarray_2d,
        axis_ranges=AXIS_BOUNDS,
    )
    figure = plot_structures(
        bounded_slice=bounded_slice,
        step_time=step_time,
    )
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
