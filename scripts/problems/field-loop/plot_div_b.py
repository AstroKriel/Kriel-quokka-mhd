## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

from matplotlib import figure as mpl_figure
from matplotlib import patches as mpl_patches
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

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class Slice:
    step_time: float
    sarray_2d: numpy_typing.NDArray[numpy.floating]


@dataclass(frozen=True)
class ProblemSetup:
    direction: tuple[float, float]
    period: float
    radius: float


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/field-loop/num_cells=96/q26-b25-ppm_ep" / "extracted"
FIGURE_PATH = ROOT_DIR / "figures/problems/field-loop/div-b.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-1.5, 1.5), (-1.0, 1.0))

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    file_glob: str,
    target_time: float,
) -> Path:
    """Return the saved slice nearest `target_time`."""
    slice_paths = sorted(DATASET_DIR.glob(file_glob))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{file_glob}` found in: {DATASET_DIR}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_slice(
    *,
    slice_path: Path,
) -> Slice:
    """Load the time and two-dimensional field from one saved slice."""
    with numpy.load(slice_path) as data:
        return Slice(
            step_time=float(data["step_time"]),
            sarray_2d=data["sarray_2d"],
        )


def compute_symmetric_bounds(
    *,
    field: numpy_typing.NDArray[numpy.floating],
) -> tuple[float, float]:
    """Return symmetric bounds set by a percentile of the field magnitude."""
    bound = float(
        numpy.nanpercentile(
            numpy.abs(field),
            99.9,
        ),
    )
    return (-bound, bound)


def compute_log10_absolute_divb(
    *,
    sarray_2d: numpy_typing.NDArray[numpy.floating],
) -> numpy_typing.NDArray[numpy.floating]:
    """Drop zero/non-finite cells (outside the loop), then take log10 of the magnitude."""
    nonzero_finite = numpy.isfinite(sarray_2d) & (sarray_2d != 0.0)
    return numpy.log10(numpy.abs(sarray_2d[nonzero_finite]))


def compute_loop_center_at_time(
    *,
    problem_setup: ProblemSetup,
    step_time: float,
) -> tuple[float, float]:
    """Return the loop's true advected centre at `step_time`, wrapped into the periodic domain."""

    def wrap(
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        span = bounds[1] - bounds[0]
        return (value - bounds[0]) % span + bounds[0]

    loop_initial_center = (0.0, 0.0)
    raw_x = loop_initial_center[0] + problem_setup.direction[0] * step_time
    raw_y = loop_initial_center[1] + problem_setup.direction[1] * step_time
    return (wrap(raw_x, AXIS_BOUNDS[0]), wrap(raw_y, AXIS_BOUNDS[1]))


def add_advection_arrow(
    *,
    panel: manage_figure.Panel,
    problem_setup: ProblemSetup,
    loop_center: tuple[float, float],
) -> None:
    figure_params = style_figure.get_figure_params()
    artist_params = figure_params.artist_params
    text_size_params = figure_params.text_size_params
    arrow_start = (
        loop_center[0] + problem_setup.radius * problem_setup.direction[0],
        loop_center[1] + problem_setup.radius * problem_setup.direction[1],
    )
    arrow_length = 0.35
    arrow_end = (
        arrow_start[0] + arrow_length * problem_setup.direction[0],
        arrow_start[1] + arrow_length * problem_setup.direction[1],
    )
    panel.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "red",
            "linewidth": artist_params.line_width_pt,
            "mutation_scale": text_size_params.annotation_size_pt,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    panel.text(
        arrow_start[0] + 0.025,
        arrow_start[1] + 0.075,
        "advection",
        ha="left",
        va="bottom",
        color="red",
        rotation=180 / numpy.pi * numpy.atan(2 / 3),
        rotation_mode="anchor",
        fontsize=text_size_params.annotation_size_pt,
    )


##
## === PANEL FUNCTIONS
##


def plot_pdf_panel(
    *,
    panel: manage_figure.Panel,
    problem_setup: ProblemSetup,
    divb_series: tuple[Slice, ...],
) -> None:
    num_pdf_bins = 50
    num_pdf_times = 10
    figure_params = style_figure.get_figure_params()
    finite_slices = [
        divb_slice for divb_slice in divb_series
        if numpy.any(numpy.isfinite(divb_slice.sarray_2d) & (divb_slice.sarray_2d != 0.0))
    ]
    sampled_indices = numpy.linspace(0, len(finite_slices) - 1, num_pdf_times, dtype=int)
    sampled_slices = [finite_slices[index] for index in sampled_indices]
    pdf_bin_edges = numpy.linspace(-70, -10, num_pdf_bins + 1)
    pdf_bin_centers = 0.5 * (pdf_bin_edges[:-1] + pdf_bin_edges[1:])
    time_palette = add_color.make_palette(
        config=add_color.SequentialConfig(
            palette_name="cmr.bubblegum_r",
            palette_range=(0.05, 0.85),
        ),
        value_range=(
            divb_series[1].step_time / problem_setup.period,
            divb_series[-1].step_time / problem_setup.period,
        ),
    )
    for sample_slice in sampled_slices[1:]:
        log10_absolute_divb = compute_log10_absolute_divb(sarray_2d=sample_slice.sarray_2d)
        estimated_pdf = compute_array_stats.estimate_pdf(
            values=log10_absolute_divb,
            bin_centers=pdf_bin_centers,
        )
        log10_pdf = numpy.ma.log10(
            numpy.ma.masked_less_equal(
                estimated_pdf.densities,
                0.0,
            ),
        )
        finite_pdf = numpy.isfinite(log10_pdf)
        curve_color = time_palette.mpl_cmap(time_palette.mpl_norm(sample_slice.step_time / problem_setup.period))
        panel.step(
            estimated_pdf.bin_centers[finite_pdf],
            log10_pdf[finite_pdf],
            where="mid",
            color=curve_color,
        )
    panel.set_xlabel(r"$x \equiv \log_{10}|\nabla \cdot \vec{b}|$")
    panel.set_ylabel(r"$\log_{10}\!\left(\mathrm{PDF}(x)\right)$")
    panel.set_xlim(-54, -13)
    panel.set_ylim(-2.3, 0.0)
    panel.yaxis.set_label_position("right")
    panel.yaxis.tick_right()
    axis_width = AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]
    axis_height = AXIS_BOUNDS[1][1] - AXIS_BOUNDS[1][0]
    panel.set_box_aspect(axis_height / axis_width)
    panel_gaps = figure_params.figure_layout.panel_gaps
    frame_params = figure_params.frame_params
    add_color.add_colorbar(
        panels=panel,
        palette=time_palette,
        label=r"$t / T$",
        colorbar_side="top",
        colorbar_gap_pt=panel_gaps.row_pt / 2.0,
        label_gap_pt=frame_params.axis_label_gap_pt * 2.0,
    )


def plot_slice_panel(
    *,
    panel: manage_figure.Panel,
    problem_setup: ProblemSetup,
    divb_slice: Slice,
) -> None:
    """Plot the div-b slice nearest the chosen target time, with a colorbar and a time label."""
    figure_params = style_figure.get_figure_params()
    artist_params = figure_params.artist_params
    frame_params = figure_params.frame_params
    panel_gaps = figure_params.figure_layout.panel_gaps
    palette_config = add_color.DivergingConfig(
        mid_value=0.0,
        palette_name="cmr.prinsenvlag",
        palette_range=(0.15, 0.85),
    )
    scaled_field = divb_slice.sarray_2d / 1.0e-16
    cbar_bounds = compute_symmetric_bounds(field=scaled_field)
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=scaled_field,
        data_format="xy",
        axis_ranges=AXIS_BOUNDS,
        colorbar_range=cbar_bounds,
        palette_config=palette_config,
        add_colorbar=False,
    )
    palette = add_color.make_palette(
        config=palette_config,
        value_range=cbar_bounds,
    )
    add_color.add_colorbar(
        panels=panel,
        palette=palette,
        label=r"$10^{16} \ (\nabla \cdot \vec{b})$",
        colorbar_side="top",
        colorbar_gap_pt=panel_gaps.row_pt / 2.0,
        label_gap_pt=frame_params.axis_label_gap_pt * 2.0,
    )
    loop_center = compute_loop_center_at_time(problem_setup=problem_setup, step_time=divb_slice.step_time)
    panel.add_patch(
        mpl_patches.Circle(
            loop_center,
            problem_setup.radius,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=artist_params.line_width_pt,
        ),
    )
    amr_region_bounds: plot_data.AxisRanges = ((-1.25, -0.75), (-0.75, 0.75))
    (amr_x_lo, amr_x_hi), (amr_y_lo, amr_y_hi) = amr_region_bounds
    panel.add_patch(
        mpl_patches.Rectangle(
            (amr_x_lo, amr_y_lo),
            amr_x_hi - amr_x_lo,
            amr_y_hi - amr_y_lo,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            linewidth=artist_params.line_width_pt,
        ),
    )
    add_advection_arrow(
        panel=panel,
        problem_setup=problem_setup,
        loop_center=loop_center,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.15,
        y_pos_fraction=0.65,
        label=r"\shortstack{refinement\\region}",
        rotate_deg=90.0,
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Center.Center,
        text_color="blue",
    )
    panel.set_xlabel(r"$x_0$")
    panel.set_ylabel(r"$x_1$")
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.5,
        y_pos_fraction=0.05,
        label=rf"$t / T = {divb_slice.step_time / problem_setup.period:.2f}$",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Bottom,
    )


def plot_field_loop_divb(
    *,
    problem_setup: ProblemSetup,
    divb_series: tuple[Slice, ...],
    divb_slice: Slice,
) -> mpl_figure.Figure:
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=1,
        num_panel_cols=2,
        panel_aspect_ratio=3.0 / 2.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.95),
        ),
    )
    plot_slice_panel(
        panel=panel_grid[0, 0],
        problem_setup=problem_setup,
        divb_slice=divb_slice,
    )
    plot_pdf_panel(
        panel=panel_grid[0, 1],
        problem_setup=problem_setup,
        divb_series=divb_series,
    )
    return figure


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    divb_glob = "magnetic_divergence-slice=x_2-index=*.npz"
    divb_paths = sorted(DATASET_DIR.glob(divb_glob))
    if not divb_paths:
        raise FileNotFoundError(f"no slice matching `{divb_glob}` found in: {DATASET_DIR}")
    divb_series = tuple(load_slice(slice_path=divb_path) for divb_path in divb_paths)
    divb_slice = load_slice(
        slice_path=find_slice_near_time(
            file_glob=divb_glob,
            target_time=2.4155,
        ),
    )
    advection_angle = numpy.arctan2(3.0, 2.0)
    problem_setup = ProblemSetup(
        direction=(numpy.sin(advection_angle), numpy.cos(advection_angle)),
        period=(AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / numpy.sin(advection_angle),
        radius=0.5,
    )
    figure = plot_field_loop_divb(
        problem_setup=problem_setup,
        divb_series=divb_series,
        divb_slice=divb_slice,
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
