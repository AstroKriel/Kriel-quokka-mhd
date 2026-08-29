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

from numpy import typing as numpy_typing

## personal
from jormi.ww_arrays import mask_2d_arrays
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    manage_figure,
    plot_data,
    style_figure,
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
    FS17 = EMFComputeSchemeStyle(label="FS18")
    B25 = EMFComputeSchemeStyle(label="B25a")
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
        label="B25b",
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
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/orszag-tang/num_cells=1024"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/orszag-tang/ncells=1024/emf-scheme-comparison.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-0.5, 0.5), (-0.5, 0.5))

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    sim_dir: Path,
    target_time: float,
) -> Path:
    file_name_glob = "current_density_magnitude-slice=x_2-index=*.npz"
    slice_paths = sorted((sim_dir / "extracted").glob(file_name_glob))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{file_name_glob}` found in: {sim_dir / 'extracted'}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - target_time),
    )


def load_log10_sarray_slice(
    *,
    sim: Simulation,
    target_time: float,
) -> numpy_typing.NDArray[numpy.floating]:
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
    sarray: numpy_typing.NDArray[numpy.floating],
    mask: numpy_typing.NDArray[numpy.bool],
) -> numpy_typing.NDArray[numpy.floating]:
    return numpy.where(mask, sarray, numpy.nan)


def plot_comparison_contours(
    *,
    panel: manage_figure.Panel,
    upper_sarray: numpy_typing.NDArray[numpy.floating],
    lower_sarray: numpy_typing.NDArray[numpy.floating],
    contour_value: float,
    upper_color: str,
    lower_color: str,
) -> None:
    """
    Overlay contours of two averaging schemes; split across the off-diagonal, each with a
    faint 'ghost' of the other scheme overlayed.
    """
    num_rows, num_cols = upper_sarray.shape
    upper_mask = mask_2d_arrays.DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    lower_mask = mask_2d_arrays.DiagonalMasks2D.get_mask_below_main_diagonal(
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
    panel.contour(
        grid_x,
        grid_y,
        upper_sarray_main.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.5,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    panel.contour(
        grid_x,
        grid_y,
        lower_sarray_main.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.5,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    panel.contour(
        grid_x,
        grid_y,
        upper_sarray_ghost.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.45,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    panel.contour(
        grid_x,
        grid_y,
        lower_sarray_ghost.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.45,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    panel.plot(
        [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
        [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
        color="black",
        linewidth=0.4,
        zorder=3,
    )


def add_label(
    *,
    panel: manage_figure.Panel,
    x_position: float,
    y_position: float,
    x_alignment: box_positions.Positions.PositionLike,
    y_alignment: box_positions.Positions.PositionLike,
    label: str,
) -> None:
    x_anchor = validate_box_positions.as_mpl_ha(x_alignment)
    y_anchor = validate_box_positions.as_mpl_va(y_alignment)
    panel.text(
        x_position,
        y_position,
        label,
        transform=panel.transAxes,
        ha=x_anchor.value,
        va=y_anchor.value,
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
    style_figure.set_figure_params()
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    target_time = 0.85
    contour_log10_value = -1.6
    reconstruction_schemes = list(ReconstructionScheme)
    emf_compute_schemes = list(EMFComputeScheme)
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=len(reconstruction_schemes),
        num_panel_cols=len(emf_compute_schemes),
        panel_aspect_ratio=1.0,
        panel_row_gap_pt=4.0,
        panel_col_gap_pt=4.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.95),
        ),
        share_x_axis=True,
        share_y_axis=True,
    )
    for row_index, reconstruction_scheme in enumerate(reconstruction_schemes):
        for col_index, emf_compute_scheme in enumerate(emf_compute_schemes):
            panel = panel_grid[row_index, col_index]
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
                target_time=target_time,
            )
            b25_log10_sarray_slice = load_log10_sarray_slice(
                sim=b25_sim,
                target_time=target_time,
            )
            plot_comparison_contours(
                panel=panel,
                upper_sarray=ld04_log10_sarray_slice,
                lower_sarray=b25_log10_sarray_slice,
                contour_value=contour_log10_value,
                upper_color=EMFAveragingScheme.LD04.value.color,
                lower_color=EMFAveragingScheme.B25.value.color,
            )
            panel.set_facecolor("white")
            panel.set_xlim(AXIS_BOUNDS[0])
            panel.set_ylim(AXIS_BOUNDS[1])
            panel.set_xticks([])
            panel.set_yticks([])
            add_label(
                panel=panel,
                x_position=0.035,
                y_position=0.965,
                x_alignment=box_positions.Positions.Side.Left,
                y_alignment=box_positions.Positions.Side.Top,
                label=EMFAveragingScheme.LD04.value.label,
            )
            add_label(
                panel=panel,
                x_position=0.965,
                y_position=0.035,
                x_alignment=box_positions.Positions.Side.Right,
                y_alignment=box_positions.Positions.Side.Bottom,
                label=EMFAveragingScheme.B25.value.label,
            )
            if row_index == 0:
                panel.set_title(emf_compute_scheme.value.label)
            if col_index == 0:
                panel.set_ylabel(reconstruction_scheme.value.label)
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
