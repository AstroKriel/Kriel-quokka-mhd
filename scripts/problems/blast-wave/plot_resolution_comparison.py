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
from jormi.ww_arrays.mask_2d_arrays import DiagonalMasks2D
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class DensitySlice:
    step_time: float
    log10_density: NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/blast-wave"
FIGURE_PATH = ROOT_DIR / "figures/problems/blast-wave/resolution-comparison.png"
## fixed recommended scheme (Q26 + Balsara2025b, PPM-EP); only resolution varies, so each split
## shows whether the resolved shock structure is grid-converged and free of artefacts. Consecutive
## pairs (rather than every combination) form a convergence ladder: each panel is its own diagonal
## split, so three resolutions need two panels, not one -- the exact mirror symmetry only supports
## a two-way split per panel.
NCELLS_VALUES = (128, 512)
NCELLS_PAIRS = tuple(
    zip(
        NCELLS_VALUES[:-1],
        NCELLS_VALUES[1:],
    ),
)
TARGET_TIME = 0.05

## plotting details
## normalising by the ambient background density (rho_0 = 1 in the initial condition) makes the
## plotted quantity read as a compression ratio relative to the undisturbed medium, rather than
## an arbitrary-unit density value
BACKGROUND_DENSITY = 1.0
FIELD_LABEL = r"$\log_{10}(\rho / \rho_0)$"
PALETTE_NAME = "cmr.copper_s"
## the computational domain is a cube [-0.5, 0.5]^3 in dimensionless units; we slice its midplane
AXIS_BOUNDS: plot_data.AxisBounds = ((-0.5, 0.5), (-0.5, 0.5))
MAJOR_TICK_STEP = 0.25
MINOR_TICK_STEP = 0.05
LABELED_TICK_VALUES = (-0.25, 0.25)

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    ncells: int,
    target_time: float,
) -> Path:
    """Return the saved density slice for `q26-b25-ppm_ep` at `ncells` nearest `target_time`."""
    extracted_dir = DATASET_DIR / f"ncells={ncells}" / "q26-b25-ppm_ep" / "extracted"
    slice_paths = sorted(extracted_dir.glob("density-slice=x_2-index=*.npz"))
    if not slice_paths:
        raise FileNotFoundError(f"no density slice found in: {extracted_dir}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_density_slice(
    *,
    ncells: int,
) -> DensitySlice:
    """Load the density slice nearest `TARGET_TIME` for one resolution."""
    slice_path = find_slice_near_time(
        ncells=ncells,
        target_time=TARGET_TIME,
    )
    with numpy.load(slice_path) as data:
        return DensitySlice(
            step_time=float(data["step_time"]),
            log10_density=numpy.log10(data["sarray_2d"] / BACKGROUND_DENSITY),
        )


def upsample_to_match(
    *,
    coarse_array: NDArray[numpy.floating],
    fine_array: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """Block-replicate `coarse_array` so its shape matches `fine_array`'s (an integer multiple).

    Nearest-neighbour replication (rather than interpolation) is deliberate: it keeps every
    coarse cell visually blocky at the fine grid's pixel scale, so the resolution difference
    between the two panels stays visible rather than being smoothed away.
    """
    factor = fine_array.shape[0] // coarse_array.shape[0]
    return numpy.kron(coarse_array, numpy.ones((factor, factor)))


def compose_resolution_split(
    *,
    upper_array: NDArray[numpy.floating],
    lower_array: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """Combine two resolutions' density fields into one array, split across the main diagonal.

    The blast-wave initial condition (a spherical overpressure) and background field
    b = (b, b, 0) are both invariant under swapping x_0 and x_1, so the converged solution
    satisfies rho(x_0, x_1) = rho(x_1, x_0): each run's own field is already mirror-symmetric
    about the main diagonal, so one triangular half per run is sufficient to show it in full,
    freeing the other half to show the other resolution directly alongside it.
    """
    if upper_array.shape != lower_array.shape:
        upper_array = upsample_to_match(
            coarse_array=upper_array,
            fine_array=lower_array,
        )
    num_rows, num_cols = upper_array.shape
    upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return numpy.where(upper_mask, upper_array, lower_array)


def format_domain_tick(
    tick_value: float,
    _tick_position: int,
) -> str:
    """Label only `LABELED_TICK_VALUES`; every other major tick is drawn unlabeled."""
    is_labeled = any(numpy.isclose(tick_value, labeled_value) for labeled_value in LABELED_TICK_VALUES)
    return f"{tick_value:.2f}" if is_labeled else ""


def configure_domain_ticks(
    *,
    ax: manage_plots.PlotAxis,
    label_left: bool,
) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(MAJOR_TICK_STEP))
        axis.set_minor_locator(MultipleLocator(MINOR_TICK_STEP))
        axis.set_major_formatter(FuncFormatter(format_domain_tick))
    ax.tick_params(
        which="both",
        color="white",
        labelbottom=True,
        labeltop=False,
        labelleft=label_left,
        labelright=False,
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
    density_slices = {ncells: load_density_slice(ncells=ncells) for ncells in NCELLS_VALUES}
    shared_value_range = (
        min(density_slice.log10_density.min() for density_slice in density_slices.values()),
        max(density_slice.log10_density.max() for density_slice in density_slices.values()),
    )
    palette_config = add_color.SequentialConfig(
        palette_name=PALETTE_NAME,
        palette_range=(0.0, 1.0),
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=len(NCELLS_PAIRS),
        axis_shape=(6, 6),
        x_spacing=0.05,
    )
    for panel_index, (ncells_upper, ncells_lower) in enumerate(NCELLS_PAIRS):
        ax = axs[0, panel_index]
        composite = compose_resolution_split(
            upper_array=density_slices[ncells_upper].log10_density,
            lower_array=density_slices[ncells_lower].log10_density,
        )
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=composite,
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=shared_value_range,
            palette_config=palette_config,
            add_cbar=False,
        )
        ax.plot(
            [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
            [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
            color="white",
            linewidth=0.6,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.05,
            y_pos=0.95,
            label=rf"${ncells_upper}^3$",
            x_alignment=box_positions.Positions.Side.Left,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=26,
            text_color="white",
            box_alpha=0.0,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.95,
            y_pos=0.05,
            label=rf"${ncells_lower}^3$",
            x_alignment=box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Bottom,
            text_size=26,
            text_color="white",
            box_alpha=0.0,
        )
        configure_domain_ticks(
            ax=ax,
            label_left=(panel_index == 0),
        )
    palette = add_color.make_palette(
        config=palette_config,
        value_range=shared_value_range,
    )
    add_color.add_colorbar(
        ax=axs[0, -1],
        palette=palette,
        label=FIELD_LABEL,
        cbar_side="top",
        label_size=26,
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
