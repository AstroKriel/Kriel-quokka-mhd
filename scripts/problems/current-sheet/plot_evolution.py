## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from matplotlib.ticker import FuncFormatter, MultipleLocator
from numpy.typing import NDArray

## personal
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_panel, manage_figure, plot_data, style_figure
from jormi.ww_types import box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class DataSlice:
    step_time: float
    current_density: NDArray[numpy.floating]


@dataclass(frozen=True)
class DataPanel:
    data_slice: DataSlice
    data_range: tuple[float, float]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/current-sheet/ncells=1024/q26-b25-ppm"
DATASET_GLOB = "current_density-comp=x_2-slice=x_2-index=*.npz"
TARGET_TIMES = (0.0, 0.5, 3.0, 10.0)
FIGURE_PATH = ROOT_DIR / "figures/problems/current-sheet/current-density-evolution.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-0.5, 0.5), (-0.5, 0.5))
MAJOR_TICK_STEP = 0.25
MINOR_TICK_STEP = 0.05
LABELED_TICK_VALUES = (-0.25, 0.25)

##
## === HELPER FUNCTIONS
##


def find_dataset_near_time(
    *,
    target_time: float,
) -> Path:
    extracted_dir = DATASET_DIR / "extracted"
    data_paths = sorted(extracted_dir.glob(DATASET_GLOB))
    if not data_paths:
        raise FileNotFoundError(f"no slice matching `{DATASET_GLOB}` found in: {extracted_dir}")
    return min(
        data_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_data_slices() -> list[DataSlice]:
    data_slices = []
    for target_time in TARGET_TIMES:
        data_path = find_dataset_near_time(target_time=target_time)
        with numpy.load(data_path) as dataset:
            data_slices.append(
                DataSlice(
                    step_time=float(dataset["step_time"]),
                    current_density=dataset["sarray_2d"],
                ),
            )
    return data_slices


def compute_upper_bound_value(
    *,
    slice_values: NDArray[numpy.floating],
) -> float:
    absolute_slice_values = numpy.abs(slice_values)
    upper_bound_value = float(
        numpy.nanpercentile(
            absolute_slice_values,
            99.9,
        ),
    )
    if upper_bound_value == 0.0:
        upper_bound_value = float(numpy.nanmax(absolute_slice_values))
    return upper_bound_value


def compute_signed_log10(
    current_density: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    return numpy.sign(current_density) * numpy.log10(1.0 + numpy.abs(current_density))


def format_domain_tick(
    tick_value: float,
    _tick_position: int,
) -> str:
    """
    Label only `LABELED_TICK_VALUES`; every other major tick is drawn unlabeled.

    Labels are math mode, so their minus signs match the ones Matplotlib formats itself.
    """
    is_labeled = any(numpy.isclose(tick_value, labeled_value) for labeled_value in LABELED_TICK_VALUES)
    return f"${tick_value:.2f}$" if is_labeled else ""


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    figure_params = style_figure.get_figure_params()
    frame_params = figure_params.frame_params
    text_size_params = figure_params.text_size_params
    theme_params = figure_params.theme_params
    ## the panels share both axes, so only their frames sit in the gaps
    panel_gaps = style_figure.PanelGaps(
        row_pt=5.0,
        col_pt=5.0,
    )
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    data_slices = load_data_slices()
    data_panels = [
        DataPanel(
            data_slice=data_slice,
            data_range=(
                float(numpy.nanmin(data_slice.current_density)),
                float(numpy.nanmax(data_slice.current_density)),
            ),
        ) for data_slice in data_slices
    ]
    palette_config = add_color.DivergingConfig(
        mid_value=0.0,
        palette_name="pink-white-green",
    )
    num_rows = 2
    num_cols = 2
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=num_rows,
        num_panel_cols=num_cols,
        ## the domain is square; the figure is fitted around it once its labels exist
        panel_aspect_ratio=1.0,
        ## drawn at the width the paper prints it at, so its text is the size it asks for
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
            panel_gaps=panel_gaps,
        ),
    )
    shared_upper_bound_value = max(
        compute_upper_bound_value(slice_values=data_slice.current_density) for data_slice in data_slices
    )
    shared_log10_upper_bound_value = float(numpy.log10(1.0 + shared_upper_bound_value))
    shared_log10_bounds = (-shared_log10_upper_bound_value, shared_log10_upper_bound_value)
    for panel_index, data_panel in enumerate(data_panels):
        row_index, col_index = divmod(panel_index, num_cols)
        panel = panel_grid[row_index, col_index]
        plot_data.plot_2d_array(
            panel=panel,
            array_2d=compute_signed_log10(data_panel.data_slice.current_density),
            data_format="xy",
            axis_ranges=AXIS_BOUNDS,
            colorbar_range=shared_log10_bounds,
            palette_config=palette_config,
            add_colorbar=False,
        )
        annotate_panel.add_text(
            panel=panel,
            x_pos_fraction=0.5,
            y_pos_fraction=0.95,
            label=rf"$t = {data_panel.data_slice.step_time:.1f}$",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
        )
        panel.text(
            0.9,
            0.5,
            rf"$j_2 \in [{data_panel.data_range[0]:.1f},\ {data_panel.data_range[1]:.1f}]$",
            transform=panel.transAxes,
            rotation=90.0,
            rotation_mode="anchor",
            horizontalalignment="center",
            verticalalignment="center",
            ## it runs the full height of a panel, so it is set tighter than the rest; usetex
            ## picks from discrete design sizes, so nearby sizes do not all shrink it
            fontsize=text_size_params.annotation_size_pt - 2.0,
            color=theme_params.foreground_color,
        )
        for axis in (panel.xaxis, panel.yaxis):
            axis.set_major_locator(MultipleLocator(MAJOR_TICK_STEP))
            axis.set_minor_locator(MultipleLocator(MINOR_TICK_STEP))
            axis.set_major_formatter(FuncFormatter(format_domain_tick))
        panel.tick_params(
            labelbottom=(row_index == num_rows - 1),
            labelleft=(col_index == 0),
        )
    palette = add_color.make_palette(
        config=palette_config,
        value_range=shared_log10_bounds,
    )
    ## one shared colorbar and one shared label per axis, all spanning the whole grid; each
    ## is placed once the panels are fitted, so none of them is positioned by hand here
    add_color.add_colorbar(
        panels=panel_grid,
        palette=palette,
        label=r"$\mathrm{sgn}(j_2)\,\log_{10}\!\left(1 + |j_2|\right)$",
        colorbar_side="top",
        ## the gap is left unset, so the bar sits off the grid by the same gap the panels
        ## are spaced by
        ## the label clears a row of tick labels here, not just the bar, so it sits further out
        label_gap_pt=frame_params.axis_label_gap_pt * 2.0,
    )
    annotate_panel.add_shared_axis_label(
        panels=panel_grid,
        label=r"$x_0$",
        side=box_positions.Positions.Side.Bottom,
    )
    annotate_panel.add_shared_axis_label(
        panels=panel_grid,
        label=r"$x_1$",
        side=box_positions.Positions.Side.Left,
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
