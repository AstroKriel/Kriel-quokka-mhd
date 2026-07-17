## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy
from matplotlib import gridspec as mpl_gridspec
from matplotlib import pyplot as mpl_plot

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

TARGET_TIME = 1.5
PERCENTILE_BOUND = 99.9
NUM_PDF_BINS = 50
PDF_BOUNDS = (-70.0, -10.0)
NUM_PDF_TIMES = 10
TICK_LABEL_SIZE = 20
AXIS_LABEL_SIZE = 25

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/field-loop/ncells=128/q26-b25-ppm_ep"
EXTRACTED_DIR = DATASET_DIR / "extracted"
FIGURE_PATH = ROOT_DIR / "figures/problems/field-loop/field-loop.png"

AXIS_BOUNDS = ((-1.0, 1.0), (-0.578125, 0.578125))

## one full advection period: the time for the loop to cross the domain once along x_0, at the
## advection velocity v_0 = sin(pi/3) set by the problem generator (testFieldLoop.cpp)
ADVECTION_PERIOD = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / numpy.sin(numpy.pi / 3.0)

DIVB_GLOB = "magnetic_divergence-slice=x_2-index=*.npz"
MAGNETIC_ENERGY_GLOB = "magnetic_energy-slice=x_2-index=*.npz"

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    file_glob: str,
    target_time: float,
) -> Path:
    """Return the saved slice nearest `target_time`."""
    slice_paths = sorted(EXTRACTED_DIR.glob(file_glob))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{file_glob}` found in: {EXTRACTED_DIR}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_slice(
    *,
    slice_path: Path,
) -> tuple[float, numpy.ndarray]:
    """Load the time and two-dimensional field from one saved slice."""
    with numpy.load(slice_path) as data:
        return float(data["step_time"]), data["sarray_2d"]


def compute_symmetric_bounds(
    *,
    field: numpy.ndarray,
) -> tuple[float, float]:
    """Return symmetric bounds set by a percentile of the field magnitude."""
    bound = float(numpy.nanpercentile(numpy.abs(field), PERCENTILE_BOUND))
    return (-bound, bound)


def plot_slice(
    *,
    ax: mpl_plot.Axes,
    field: numpy.ndarray,
    cbar_bounds: tuple[float, float],
    palette_config: add_color.SequentialConfig | add_color.DivergingConfig,
    cbar_label: str,
) -> None:
    """Plot one field-loop slice with a separate horizontal colorbar."""
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=field,
        data_format="xy",
        axis_bounds=AXIS_BOUNDS,
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
        label=cbar_label,
        cbar_side="right",
        cbar_thickness=0.065,
        cbar_pad=0.02,
        label_size=AXIS_LABEL_SIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()

    divb_paths = sorted(EXTRACTED_DIR.glob(DIVB_GLOB))
    if not divb_paths:
        raise FileNotFoundError(f"no slice matching `{DIVB_GLOB}` found in: {EXTRACTED_DIR}")
    divb_series = [load_slice(slice_path=divb_path) for divb_path in divb_paths]

    divb_time, divb = load_slice(
        slice_path=find_slice_near_time(file_glob=DIVB_GLOB, target_time=TARGET_TIME),
    )
    magnetic_energy_time, magnetic_energy = load_slice(
        slice_path=find_slice_near_time(file_glob=MAGNETIC_ENERGY_GLOB, target_time=TARGET_TIME),
    )
    if not numpy.isclose(divb_time, magnetic_energy_time):
        raise ValueError(
            "divergence and magnetic-energy slices do not correspond to the same time: "
            f"{divb_time} != {magnetic_energy_time}",
        )

    fig = mpl_plot.figure(figsize=(7.0, 12.25))
    grid_spec = mpl_gridspec.GridSpec(
        nrows=3,
        ncols=1,
        figure=fig,
        height_ratios=(1.0, 1.0, 1.0),
        hspace=0.10,
    )
    ax_magnetic_energy = fig.add_subplot(grid_spec[0, 0])
    ax_divb = fig.add_subplot(grid_spec[1, 0])
    ax_pdf = fig.add_subplot(grid_spec[2, 0])

    log10_absolute_divb_series = [
        (
            step_time,
            numpy.log10(numpy.abs(step_divb[numpy.isfinite(step_divb) & (step_divb != 0.0)])),
        )
        for step_time, step_divb in divb_series
        if numpy.any(numpy.isfinite(step_divb) & (step_divb != 0.0))
    ]
    sampled_time_indices = numpy.linspace(
        0,
        len(log10_absolute_divb_series) - 1,
        NUM_PDF_TIMES,
        dtype=int,
    )
    log10_absolute_divb_series = [
        log10_absolute_divb_series[index] for index in sampled_time_indices
    ]
    pdf_bin_edges = numpy.linspace(PDF_BOUNDS[0], PDF_BOUNDS[1], NUM_PDF_BINS + 1)
    pdf_bin_centers = 0.5 * (pdf_bin_edges[:-1] + pdf_bin_edges[1:])
    time_palette = add_color.make_palette(
        config=add_color.SequentialConfig(
            palette_name="cmr.bubblegum",
            palette_range=(0.15, 0.95),
        ),
        value_range=(
            divb_series[0][0] / ADVECTION_PERIOD,
            divb_series[-1][0] / ADVECTION_PERIOD,
        ),
    )
    for step_time, log10_absolute_divb in log10_absolute_divb_series:
        estimated_pdf = compute_array_stats.estimate_pdf(
            values=log10_absolute_divb,
            bin_centers=pdf_bin_centers,
        )
        log10_pdf = numpy.ma.log10(
            numpy.ma.masked_less_equal(estimated_pdf.densities, 0.0),
        )
        finite_pdf = numpy.isfinite(log10_pdf)
        curve_color = time_palette.mpl_cmap(time_palette.mpl_norm(step_time / ADVECTION_PERIOD))
        ax_pdf.step(
            estimated_pdf.bin_centers[finite_pdf],
            log10_pdf[finite_pdf],
            where="mid",
            color=curve_color,
            linewidth=2.0,
        )
    ax_pdf.set_xlabel(
        r"$x \equiv \log_{10}|\nabla \cdot \vec{b}|$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax_pdf.set_ylabel(
        r"$\log_{10}\!\left(\mathrm{PDF}(x)\right)$",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax_pdf.set_ylim(-2.5, 0.0)
    ax_pdf.tick_params(labelsize=TICK_LABEL_SIZE)
    ax_pdf.yaxis.set_label_position("left")
    ax_pdf.yaxis.tick_left()
    axis_width = AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]
    axis_height = AXIS_BOUNDS[1][1] - AXIS_BOUNDS[1][0]
    ax_pdf.set_box_aspect(axis_height / axis_width)
    time_cbar = add_color.add_colorbar(
        ax=ax_pdf,
        palette=time_palette,
        label=r"$t / T_\mathrm{advect}$",
        cbar_side="right",
        cbar_thickness=0.065,
        cbar_pad=0.02,
        label_size=AXIS_LABEL_SIZE,
    )
    time_cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    divb_palette_config = add_color.DivergingConfig(
        mid_value=0.0,
        palette_name="bwr",
    )
    plot_slice(
        ax=ax_divb,
        field=divb / 1.0e-16,
        cbar_bounds=compute_symmetric_bounds(field=divb / 1.0e-16),
        palette_config=divb_palette_config,
        cbar_label=r"$(\nabla \cdot \vec{b}) / 10^{-16}$",
    )

    magnetic_energy_palette_config = add_color.SequentialConfig(
        palette_name="cmr.horizon_r",
    )
    log10_magnetic_energy = compute_array_stats.compute_safe_log10(magnetic_energy)
    magnetic_energy_upper_bound = float(
        numpy.nanpercentile(log10_magnetic_energy, PERCENTILE_BOUND),
    )
    magnetic_energy_bounds = (magnetic_energy_upper_bound - 6.0, magnetic_energy_upper_bound)
    plot_slice(
        ax=ax_magnetic_energy,
        field=log10_magnetic_energy,
        cbar_bounds=magnetic_energy_bounds,
        palette_config=magnetic_energy_palette_config,
        cbar_label=r"$\log_{10}(b^2/2)$",
    )
    for ax in (ax_magnetic_energy, ax_divb):
        ax.set_ylabel(r"$x_1$", fontsize=AXIS_LABEL_SIZE)
    ax_magnetic_energy.set_xlabel(r"$x_0$", fontsize=AXIS_LABEL_SIZE)
    ax_magnetic_energy.xaxis.set_label_position("top")
    ax_magnetic_energy.xaxis.tick_top()
    ax_magnetic_energy.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax_divb.tick_params(axis="x", labelbottom=False)
    for ax in (ax_magnetic_energy, ax_divb):
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.95,
            label=rf"$t / T_\mathrm{{advect}} = {divb_time / ADVECTION_PERIOD:.2f}$",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=TICK_LABEL_SIZE,
            text_color="black",
            box_alpha=0.0,
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
