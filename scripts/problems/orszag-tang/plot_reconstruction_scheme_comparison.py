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
    reconstruction_scheme: ReconstructionScheme,
    target_time: float,
) -> numpy_typing.NDArray[numpy.floating]:
    sim_dir = DATASET_DIR / f"q26-b25-{reconstruction_scheme.as_tag}"
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
    plot_comparison_contours(
        panel=panel,
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
    add_label(
        panel=panel,
        x_position=0.05,
        y_position=0.95,
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        label=ReconstructionScheme.PPM.value.label,
    )
    add_label(
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
