## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
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
    ax.set_xlim(-51, -13)
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
        cbar_thickness=0.075,
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
        palette_name="bwr",
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
        label=r"$(\nabla \cdot \vec{b}) / 10^{-16}$",
        cbar_side="top",
        cbar_thickness=0.075,
        cbar_pad=0.01,
        label_size=AXIS_LABEL_SIZE,
        label_pad=17.5,
    )
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
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
