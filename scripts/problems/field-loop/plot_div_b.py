## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from matplotlib import patches as mpl_patches
from matplotlib.figure import Figure as mpl_Figure
from numpy.typing import NDArray

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class Slice:
    step_time: float
    sarray_2d: NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/field-loop/ncells=128/q26-b25-ppm_ep" / "extracted"
DIVB_GLOB = "magnetic_divergence-slice=x_2-index=*.npz"
FIGURE_PATH = ROOT_DIR / "figures/problems/field-loop/div-b.png"
TARGET_TIME = 1.5

## plotting details
NUM_PDF_BINS = 50
NUM_PDF_TIMES = 10
TICK_LABEL_SIZE = 20
AXIS_LABEL_SIZE = 25
SLICE_BOUNDS: plot_data.AxisBounds = ((-1.0, 1.0), (-0.578125, 0.578125))
ADVECTION_PERIOD = (SLICE_BOUNDS[0][1] - SLICE_BOUNDS[0][0]) / numpy.sin(numpy.pi / 3.0)

## annotations
AMR_REGION_BOUNDS: plot_data.AxisBounds = ((0.4, 0.6), (-0.23125, 0.23125))
## the loop is centred on the origin at t=0, with radius R_0=0.3 (testFieldLoop.cpp)
LOOP_INITIAL_CENTER = (0.0, 0.0)
LOOP_INITIAL_RADIUS = 0.3
## advection velocity projected onto the x_0-x_1 plane, from u=(sin(pi/3), cos(pi/3), 1)
ADVECTION_DIRECTION = (numpy.sin(numpy.pi / 3.0), numpy.cos(numpy.pi / 3.0))

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
    field: NDArray[numpy.floating],
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
    sarray_2d: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """Drop zero/non-finite cells (outside the loop), then take log10 of the magnitude."""
    nonzero_finite = numpy.isfinite(sarray_2d) & (sarray_2d != 0.0)
    return numpy.log10(numpy.abs(sarray_2d[nonzero_finite]))


def add_advection_arrow(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    arrow_start = (
        LOOP_INITIAL_CENTER[0] + LOOP_INITIAL_RADIUS * ADVECTION_DIRECTION[0],
        LOOP_INITIAL_CENTER[1] + LOOP_INITIAL_RADIUS * ADVECTION_DIRECTION[1],
    )
    arrow_length = 0.25
    arrow_end = (
        arrow_start[0] + arrow_length * ADVECTION_DIRECTION[0],
        arrow_start[1] + arrow_length * ADVECTION_DIRECTION[1],
    )
    ax.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "red",
            "linewidth": 1.5,
            "mutation_scale": 15.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    ax.text(
        arrow_start[0] + 0.025,
        arrow_start[1] + 0.05,
        "advection",
        ha="left",
        va="bottom",
        color="red",
        rotation=30.0,
        rotation_mode="anchor",
        fontsize=TICK_LABEL_SIZE,
    )


##
## === PANEL FUNCTIONS
##


def plot_pdf_panel(
    *,
    ax: manage_plots.PlotAxis,
    divb_series: tuple[Slice, ...],
) -> None:
    """Plot the div-b PDF at `NUM_PDF_TIMES` times, sampled evenly across the run."""
    finite_slices = [
        divb_slice for divb_slice in divb_series
        if numpy.any(numpy.isfinite(divb_slice.sarray_2d) & (divb_slice.sarray_2d != 0.0))
    ]
    sampled_indices = numpy.linspace(0, len(finite_slices) - 1, NUM_PDF_TIMES, dtype=int)
    sampled_slices = [finite_slices[index] for index in sampled_indices]
    pdf_bin_edges = numpy.linspace(-70, -10, NUM_PDF_BINS + 1)
    pdf_bin_centers = 0.5 * (pdf_bin_edges[:-1] + pdf_bin_edges[1:])
    time_palette = add_color.make_palette(
        config=add_color.SequentialConfig(
            palette_name="cmr.bubblegum",
            palette_range=(0.15, 0.95),
        ),
        value_range=(
            divb_series[0].step_time / ADVECTION_PERIOD,
            divb_series[-1].step_time / ADVECTION_PERIOD,
        ),
    )
    for sample_slice in sampled_slices:
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
        curve_color = time_palette.mpl_cmap(time_palette.mpl_norm(sample_slice.step_time / ADVECTION_PERIOD))
        ax.step(
            estimated_pdf.bin_centers[finite_pdf],
            log10_pdf[finite_pdf],
            where="mid",
            color=curve_color,
            linewidth=2.0,
        )
    ax.set_xlabel(
        r"$x \equiv \log_{10}|\nabla \cdot \vec{b}|$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_ylabel(
        r"$\log_{10}\!\left(\mathrm{PDF}(x)\right)$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_xlim(-52, -13)
    ax.set_ylim(-2.3, 0.0)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    axis_width = SLICE_BOUNDS[0][1] - SLICE_BOUNDS[0][0]
    axis_height = SLICE_BOUNDS[1][1] - SLICE_BOUNDS[1][0]
    ax.set_box_aspect(axis_height / axis_width)
    time_cbar = add_color.add_colorbar(
        ax=ax,
        palette=time_palette,
        label=r"$t / T_\mathrm{advect}$",
        cbar_side="top",
        cbar_thickness=0.1,
        cbar_pad=0.01,
        label_size=AXIS_LABEL_SIZE,
        label_pad=17.5,
    )
    time_cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)


def plot_slice_panel(
    *,
    ax: manage_plots.PlotAxis,
    divb_slice: Slice,
) -> None:
    """Plot the div-b slice nearest `TARGET_TIME`, with a colorbar and a time label."""
    palette_config = add_color.DivergingConfig(
        mid_value=0.0,
        palette_name="cmr.prinsenvlag",
        palette_range=(0.15, 0.85),
    )
    scaled_field = divb_slice.sarray_2d / 1.0e-16
    cbar_bounds = compute_symmetric_bounds(field=scaled_field)
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=scaled_field,
        data_format="xy",
        axis_bounds=SLICE_BOUNDS,
        cbar_bounds=cbar_bounds,
        palette_config=palette_config,
        add_cbar=False,
    )
    palette = add_color.make_palette(
        config=palette_config,
        value_range=cbar_bounds,
    )
    cbar = add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$10^{16} (\nabla \cdot \vec{b})$",
        cbar_side="top",
        cbar_thickness=0.1,
        cbar_pad=0.01,
        label_size=AXIS_LABEL_SIZE,
        label_pad=17.5,
    )
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.add_patch(
        mpl_patches.Circle(
            LOOP_INITIAL_CENTER,
            LOOP_INITIAL_RADIUS,
            fill=False,
            edgecolor="red",
            linestyle=":",
            linewidth=1.25,
        ),
    )
    (amr_x_lo, amr_x_hi), (amr_y_lo, amr_y_hi) = AMR_REGION_BOUNDS
    ax.add_patch(
        mpl_patches.Rectangle(
            (amr_x_lo, amr_y_lo),
            amr_x_hi - amr_x_lo,
            amr_y_hi - amr_y_lo,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            linewidth=1.25,
        ),
    )
    add_advection_arrow(ax=ax)
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.825,
        y_pos=0.5,
        label=r"\shortstack{refinement\\region}",
        rotate_deg=90.0,
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Center.Center,
        text_size=TICK_LABEL_SIZE,
        text_color="blue",
        box_alpha=0.0,
    )
    ax.set_xlabel(
        r"$x_0$",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=10.0,
    )
    ax.set_ylabel(r"$x_1$", fontsize=AXIS_LABEL_SIZE)
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.05,
        label=rf"$t / T_\mathrm{{advect}} = {divb_slice.step_time / ADVECTION_PERIOD:.2f}$",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Bottom,
        text_size=TICK_LABEL_SIZE,
        text_color="black",
        box_alpha=0.0,
    )


def plot_field_loop_divb(
    *,
    divb_series: tuple[Slice, ...],
    divb_slice: Slice,
) -> mpl_Figure:
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=2,
        axis_shape=(7.0, 4.5),
        fig_scale=1.5,
        x_spacing=0.05,
    )
    plot_slice_panel(
        ax=axs[0, 0],
        divb_slice=divb_slice,
    )
    plot_pdf_panel(
        ax=axs[0, 1],
        divb_series=divb_series,
    )
    return fig


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
    divb_paths = sorted(DATASET_DIR.glob(DIVB_GLOB))
    if not divb_paths:
        raise FileNotFoundError(f"no slice matching `{DIVB_GLOB}` found in: {DATASET_DIR}")
    divb_series = tuple(load_slice(slice_path=divb_path) for divb_path in divb_paths)
    divb_slice = load_slice(
        slice_path=find_slice_near_time(
            file_glob=DIVB_GLOB,
            target_time=TARGET_TIME,
        ),
    )
    fig = plot_field_loop_divb(
        divb_series=divb_series,
        divb_slice=divb_slice,
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
