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
class Slice:
    step_time: float
    current_density: NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/current-sheet/ncells=1024/q26-b25-ppm"
FIGURE_PATH = ROOT_DIR / "figures/problems/current-sheet/current-density-evolution.png"
TARGET_TIMES = (0.0, 0.5, 3.0, 10.0)
SLICE_GLOB = "current_density-comp=x_2-slice=x_2-index=*.npz"

## plotting details
FIELD_LABEL = r"$\mathrm{sign}(j_z)\,\log_{10}\!\left(1 + |j_z|\right)$"
PALETTE_NAME = "bwr"
PALETTE_RANGE = (0.0, 1.0)
PERCENTILE_BOUND = 99.9
SIGNED_LOG_SCALE = 1.0
AXIS_BOUNDS: plot_data.AxisBounds = ((-0.5, 0.5), (-0.5, 0.5))

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    target_time: float,
) -> Path:
    """Return the saved out-of-plane current-density slice nearest `target_time`."""
    extracted_dir = DATASET_DIR / "extracted"
    slice_paths = sorted(extracted_dir.glob(SLICE_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {extracted_dir}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_slices() -> list[Slice]:
    """Load the current-density slice nearest each requested physical time."""
    slices = []
    for target_time in TARGET_TIMES:
        slice_path = find_slice_near_time(target_time=target_time)
        with numpy.load(slice_path) as data:
            slices.append(
                Slice(
                    step_time=float(data["step_time"]),
                    current_density=data["sarray_2d"],
                ),
            )
    return slices


def compute_symmetric_value_range(
    *,
    current_density: NDArray[numpy.floating],
) -> tuple[float, float]:
    """Return symmetric 99.9th-percentile bounds, falling back for sparse, degenerate data."""
    absolute_current_density = numpy.abs(current_density)
    bound = float(
        numpy.nanpercentile(
            absolute_current_density,
            PERCENTILE_BOUND,
        ),
    )
    if bound == 0.0:
        bound = float(numpy.nanmax(absolute_current_density))
    return (-bound, bound)


def compute_signed_log10(
    current_density: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """Compress `current_density` while preserving its sign and a linear region near zero."""
    return numpy.sign(current_density) * numpy.log10(
        1.0 + numpy.abs(current_density) / SIGNED_LOG_SCALE,
    )


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
    slices = load_slices()
    palette_config = add_color.DivergingConfig(
        mid_value=0.0,
        palette_name=PALETTE_NAME,
        palette_range=PALETTE_RANGE,
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=2,
        num_cols=2,
        axis_shape=(3, 3),
        x_spacing=0.05,
        y_spacing=0.05,
    )
    value_ranges = [
        compute_symmetric_value_range(current_density=slice_.current_density) for slice_ in slices
    ]
    data_ranges = [
        (float(numpy.nanmin(slice_.current_density)), float(numpy.nanmax(slice_.current_density)))
        for slice_ in slices
    ]
    shared_physical_bound = max(value_range[1] for value_range in value_ranges)
    shared_log_bound = float(numpy.log10(1.0 + shared_physical_bound / SIGNED_LOG_SCALE))
    shared_log_range = (-shared_log_bound, shared_log_bound)
    panels = list(
        zip(
            slices,
            data_ranges,
            strict=True,
        ),
    )
    for panel_index, (slice_, data_range) in enumerate(panels):
        row_index, col_index = divmod(panel_index, 2)
        ax = axs[row_index, col_index]
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=compute_signed_log10(slice_.current_density),
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=shared_log_range,
            palette_config=palette_config,
            add_cbar=False,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.965,
            label=rf"$t = {slice_.step_time:.1f}$",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=20,
            text_color="black",
            box_alpha=0.0,
        )
        ax.text(
            0.86,
            0.5,
            rf"$j_z \in [{data_range[0]:.1f},\ {data_range[1]:.1f}]$",
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
        value_range=shared_log_range,
    )
    ## one shared colorbar spanning the full width of the grid (not achievable via a single
    ## panel's cbar_side placement), anchored to an invisible axis covering all four panels
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
        label=FIELD_LABEL,
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
