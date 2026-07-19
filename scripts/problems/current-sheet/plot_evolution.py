## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
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
AXIS_BOUNDS: plot_data.AxisBounds = ((-0.5, 0.5), (-0.5, 0.5))

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


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
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
        palette_name="bwr",
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=2,
        num_cols=2,
        axis_shape=(3, 3),
        x_spacing=0.05,
        y_spacing=0.05,
    )
    shared_upper_bound_value = max(
        compute_upper_bound_value(slice_values=data_slice.current_density) for data_slice in data_slices
    )
    shared_log10_upper_bound_value = float(numpy.log10(1.0 + shared_upper_bound_value))
    shared_log10_range = (-shared_log10_upper_bound_value, shared_log10_upper_bound_value)
    for panel_index, data_panel in enumerate(data_panels):
        row_index, col_index = divmod(panel_index, 2)
        ax = axs[row_index, col_index]
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=compute_signed_log10(data_panel.data_slice.current_density),
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=shared_log10_range,
            palette_config=palette_config,
            add_cbar=False,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.965,
            label=rf"$t = {data_panel.data_slice.step_time:.1f}$",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=20,
            text_color="black",
            box_alpha=0.0,
        )
        ax.text(
            0.86,
            0.5,
            rf"$j_z \in [{data_panel.data_range[0]:.1f},\ {data_panel.data_range[1]:.1f}]$",
            transform=ax.transAxes,
            rotation=90.0,
            rotation_mode="anchor",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="black",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    palette = add_color.make_palette(
        config=palette_config,
        value_range=shared_log10_range,
    )
    ## one shared colorbar spanning the full width of the grid
    top_left_bounds = axs[0, 0].get_position()
    top_right_bounds = axs[0, -1].get_position()
    bottom_left_bounds = axs[-1, 0].get_position()
    cbar_anchor_ax = fig.add_axes(
        (
            top_left_bounds.x0,
            bottom_left_bounds.y0,
            top_right_bounds.x1 - top_left_bounds.x0,
            top_left_bounds.y1 - bottom_left_bounds.y0,
        ),
    )
    cbar_anchor_ax.set_axis_off()
    add_color.add_colorbar(
        ax=cbar_anchor_ax,
        palette=palette,
        label=r"$\mathrm{sgn}(j_z)\,\log_{10}\!\left(1 + |j_z|\right)$",
        cbar_side="top",
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
