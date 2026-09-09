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
from ww_quokka_sims.sim_io.snapshots import find_snapshots

## local
from local_helpers import mask_contours, paper_style

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ReconstructionSchemeStyle:
    label: str
    color: str


class ReconstructionScheme(Enum):
    PPM = ReconstructionSchemeStyle(
        label="PPM",
        color="#DC5664",
    )
    PPM_EP = ReconstructionSchemeStyle(
        label="PPM-EP",
        color="#56DCCE",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/orszag-tang/num_cells=4096"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/orszag-tang/ncells=4096/reconstruction-scheme-comparison.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-0.5, 0.5), (-0.5, 0.5))

##
## === HELPER FUNCTIONS
##


def load_log10_sarray_slice(
    *,
    reconstruction_scheme: ReconstructionScheme,
    target_time: float,
) -> numpy_typing.NDArray[numpy.floating]:
    sim_dir = DATASET_DIR / f"q26-b25-{reconstruction_scheme.as_tag}"
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
    ppm_log10_sarray_slice = load_log10_sarray_slice(
        reconstruction_scheme=ReconstructionScheme.PPM,
        target_time=target_time,
    )
    ppm_ep_log10_sarray_slice = load_log10_sarray_slice(
        reconstruction_scheme=ReconstructionScheme.PPM_EP,
        target_time=target_time,
    )
    figure, panel = manage_figure.create_figure(
        panel_aspect_ratio=1.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.475),
        ),
    )
    mask_contours.plot_comparison_contours(
        panel=panel,
        axis_bounds=AXIS_BOUNDS,
        upper_sarray=ppm_log10_sarray_slice,
        lower_sarray=ppm_ep_log10_sarray_slice,
        contour_value=contour_log10_value,
        upper_color=ReconstructionScheme.PPM.value.color,
        lower_color=ReconstructionScheme.PPM_EP.value.color,
    )
    panel.set_facecolor("white")
    panel.set_xlim(AXIS_BOUNDS[0])
    panel.set_ylim(AXIS_BOUNDS[1])
    panel.set_xticks([])
    panel.set_yticks([])
    mask_contours.add_label(
        panel=panel,
        x_position=0.05,
        y_position=0.95,
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        label=ReconstructionScheme.PPM.value.label,
    )
    mask_contours.add_label(
        panel=panel,
        x_position=0.95,
        y_position=0.05,
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Bottom,
        label=ReconstructionScheme.PPM_EP.value.label,
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
