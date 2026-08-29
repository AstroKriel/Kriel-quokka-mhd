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
from jormi.ww_io import manage_io
from jormi.ww_plots import (
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io import find_snapshots

## local
from local_helpers import mask_contours, paper_style

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


def load_log10_sarray_slice(
    *,
    sim: Simulation,
    target_time: float,
) -> numpy_typing.NDArray[numpy.floating]:
    sim_dir = DATASET_DIR / sim.as_tag
    slice_path = find_snapshots.find_npz_near_time(
        extracted_dir=sim_dir / "extracted",
        glob_pattern="current_density_magnitude-slice=x_2-index=*.npz",
        target_time=target_time,
    )
    sarray_2d = numpy.load(slice_path)["sarray_2d"]
    cell_size = (AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / sarray_2d.shape[0]
    return numpy.log10(cell_size * sarray_2d)


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
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
            mask_contours.plot_comparison_contours(
                panel=panel,
                axis_bounds=AXIS_BOUNDS,
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
            mask_contours.add_label(
                panel=panel,
                x_position=0.035,
                y_position=0.965,
                x_alignment=box_positions.Positions.Side.Left,
                y_alignment=box_positions.Positions.Side.Top,
                label=EMFAveragingScheme.LD04.value.label,
            )
            mask_contours.add_label(
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
