## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
import numpy

from numpy.typing import NDArray

## personal
from jormi.ww_arrays.mask_2d_arrays import DiagonalMasks2D
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    manage_plots,
    plot_data,
    style_plots,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class EMFComputeSchemeStyle:
    label: str


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    label: str
    color: str


@dataclass(frozen=True)
class ReconstructionSchemeStyle:
    label: str


class EMFComputeScheme(Enum):
    FS17 = EMFComputeSchemeStyle(label="FS17")
    B25 = EMFComputeSchemeStyle(label="B25")
    Q26 = EMFComputeSchemeStyle(label="Q26")

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class EMFAveragingScheme(Enum):
    LD04 = EMFAveragingSchemeStyle(
        label="LD04",
        color="#DC5664",
    )
    B25 = EMFAveragingSchemeStyle(
        label="B25",
        color="#56DCCE",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class ReconstructionScheme(Enum):
    PLM = ReconstructionSchemeStyle(label="PLM")
    PPM = ReconstructionSchemeStyle(label="PPM")
    PPM_EP = ReconstructionSchemeStyle(label="PPM-EP")

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Simulation:
    emf_compute_scheme: EMFComputeScheme
    emf_averaging_scheme: EMFAveragingScheme
    reconstruction_scheme: ReconstructionScheme

    @property
    def as_tag(
        self,
    ) -> str:
        return (
            f"{self.emf_compute_scheme.as_tag}-"
            f"{self.emf_averaging_scheme.as_tag}-"
            f"{self.reconstruction_scheme.as_tag}"
        )


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/orszag-tang/ncells=1024"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/orszag-tang/ncells=1024/emf-scheme-comparison.png"
FILE_NAME_GLOB = "current_density_magnitude-slice=x_2-index=*.npz"
TARGET_TIME = 0.85

## plotting details
AXIS_BOUNDS: plot_data.AxisBounds = ((-0.5, 0.5), (-0.5, 0.5))
CONTOUR_LOG10_VALUE = -1.6

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    sim_dir: Path,
    target_time: float,
) -> Path:
    slice_paths = sorted((sim_dir / "extracted").glob(FILE_NAME_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{FILE_NAME_GLOB}` found in: {sim_dir / 'extracted'}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_log10_sarray_slice(
    *,
    sim: Simulation,
    target_time: float,
) -> NDArray[numpy.floating]:
    sim_dir = DATASET_DIR / sim.as_tag
    slice_path = find_slice_near_time(
        sim_dir=sim_dir,
        target_time=target_time,
    )
    sarray_2d = numpy.load(slice_path)["sarray_2d"]
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / sarray_2d.shape[0]
    return numpy.log10(cell_size * sarray_2d)


def mask_sarray_slice(
    *,
    sarray: NDArray[numpy.floating],
    mask: NDArray[numpy.bool],
) -> NDArray[numpy.floating]:
    return numpy.where(mask, sarray, numpy.nan)


def plot_comparison_contours(
    *,
    ax: manage_plots.PlotAxis,
    upper_sarray: NDArray[numpy.floating],
    lower_sarray: NDArray[numpy.floating],
    contour_value: float,
    upper_color: str,
    lower_color: str,
) -> None:
    """
    Overlay contours of two averaging schemes; split across the off-diagonal, each with a
    faint 'ghost' of the other scheme overlayed.
    """
    num_rows, num_cols = upper_sarray.shape
    upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    lower_mask = DiagonalMasks2D.get_mask_below_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    upper_sarray_main = mask_sarray_slice(
        sarray=upper_sarray,
        mask=upper_mask,
    )
    lower_sarray_main = mask_sarray_slice(
        sarray=lower_sarray,
        mask=lower_mask,
    )
    upper_sarray_ghost = mask_sarray_slice(
        sarray=upper_sarray,
        mask=lower_mask,
    )
    lower_sarray_ghost = mask_sarray_slice(
        sarray=lower_sarray,
        mask=upper_mask,
    )
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1], num_cols),
        numpy.linspace(AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1], num_rows),
    )
    ## contours of each scheme's solution
    ax.contour(
        grid_x,
        grid_y,
        upper_sarray_main.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.75,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    ax.contour(
        grid_x,
        grid_y,
        lower_sarray_main.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.75,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    ## faint "ghost" reference contours in the opposite triangle (overlayed on top)
    ax.contour(
        grid_x,
        grid_y,
        upper_sarray_ghost.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.65,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    ax.contour(
        grid_x,
        grid_y,
        lower_sarray_ghost.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.65,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    ## off-diagonal line separating the two triangular halves of the domain
    ax.plot(
        [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
        [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
        color="black",
        linewidth=0.6,
        zorder=3,
    )


def add_label(
    *,
    ax: manage_plots.PlotAxis,
    x_position: float,
    y_position: float,
    x_alignment: box_positions.Positions.PositionLike,
    y_alignment: box_positions.Positions.PositionLike,
    label: str,
) -> None:
    x_anchor = validate_box_positions.as_mpl_ha(x_alignment)
    y_anchor = validate_box_positions.as_mpl_va(y_alignment)
    ax.text(
        x_position,
        y_position,
        label,
        transform=ax.transAxes,
        ha=x_anchor.value,
        va=y_anchor.value,
        fontsize=22,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.0",
        },
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    reconstruction_schemes = list(ReconstructionScheme)
    emf_compute_schemes = list(EMFComputeScheme)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=len(reconstruction_schemes),
        num_cols=len(emf_compute_schemes),
        axis_shape=(4, 4),
        x_spacing=0.02,
        y_spacing=0.02,
        share_x=True,
        share_y=True,
    )
    for row_index, reconstruction_scheme in enumerate(reconstruction_schemes):
        for col_index, emf_compute_scheme in enumerate(emf_compute_schemes):
            ax = axs[row_index, col_index]
            ld04_sim = Simulation(
                emf_compute_scheme=emf_compute_scheme,
                emf_averaging_scheme=EMFAveragingScheme.LD04,
                reconstruction_scheme=reconstruction_scheme,
            )
            b25_sim = Simulation(
                emf_compute_scheme=emf_compute_scheme,
                emf_averaging_scheme=EMFAveragingScheme.B25,
                reconstruction_scheme=reconstruction_scheme,
            )
            ld04_log10_sarray_slice = load_log10_sarray_slice(
                sim=ld04_sim,
                target_time=TARGET_TIME,
            )
            b25_log10_sarray_slice = load_log10_sarray_slice(
                sim=b25_sim,
                target_time=TARGET_TIME,
            )
            plot_comparison_contours(
                ax=ax,
                upper_sarray=ld04_log10_sarray_slice,
                lower_sarray=b25_log10_sarray_slice,
                contour_value=CONTOUR_LOG10_VALUE,
                upper_color=EMFAveragingScheme.LD04.value.color,
                lower_color=EMFAveragingScheme.B25.value.color,
            )
            ax.set_facecolor("white")
            ax.set_xlim(AXIS_BOUNDS[0])
            ax.set_ylim(AXIS_BOUNDS[1])
            ax.set_xticks([])
            ax.set_yticks([])
            add_label(
                ax=ax,
                x_position=0.05,
                y_position=0.95,
                x_alignment=box_positions.Positions.Side.Left,
                y_alignment=box_positions.Positions.Side.Top,
                label=EMFAveragingScheme.LD04.value.label,
            )
            add_label(
                ax=ax,
                x_position=0.95,
                y_position=0.05,
                x_alignment=box_positions.Positions.Side.Right,
                y_alignment=box_positions.Positions.Side.Bottom,
                label=EMFAveragingScheme.B25.value.label,
            )
            if row_index == 0:
                ax.set_title(emf_compute_scheme.value.label)
            if col_index == 0:
                ax.set_ylabel(reconstruction_scheme.value.label)
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
