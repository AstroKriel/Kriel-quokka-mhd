## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

from numpy import typing as numpy_typing

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import annotate_panel, manage_figure, style_figure
from jormi.ww_types import box_positions

## local
from local_helpers import paper_style

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class PressureSnapshot:
    step_time: float
    pressure: numpy_typing.NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
COMBO_OFF = "q26-b25-ppm_ep-no-carbuncle-fix"
COMBO_ON = "q26-b25-ppm_ep"
ROOT_DIR = Path(__file__).parents[3]
DATASET_ROOT = ROOT_DIR / "datasets/problems/quirk/num_cells=128"
FIGURE_PATH = ROOT_DIR / "figures/problems/quirk/ncells=128/odd_even_split.png"

## annotations
EVEN_ROW_INDEX = 0
ODD_ROW_INDEX = 1

##
## === HELPER FUNCTIONS
##


def load_snapshots(
    *,
    combo: str,
) -> list[PressureSnapshot]:
    dataset_dir = DATASET_ROOT / combo / "extracted"
    pressure_paths = sorted(
        dataset_dir.glob("pressure-slice=x_2-index=*.npz"),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    snapshots = []
    for pressure_path in pressure_paths:
        with numpy.load(pressure_path) as pressure_data:
            snapshots.append(
                PressureSnapshot(
                    step_time=float(pressure_data["step_time"]),
                    pressure=pressure_data["sarray_2d"],
                ),
            )
    return snapshots


def plot_even_odd_profiles(
    *,
    panel: manage_figure.Panel,
    snapshot: PressureSnapshot,
    color: annotate_panel.ColorType,
    zorder: int,
) -> None:
    num_cells = snapshot.pressure.shape[0]
    x_0 = (numpy.arange(num_cells) + 0.5) / num_cells
    panel.plot(
        x_0,
        snapshot.pressure[:, EVEN_ROW_INDEX],
        color=color,
        linestyle="-",
        zorder=zorder,
    )
    panel.plot(
        x_0,
        snapshot.pressure[:, ODD_ROW_INDEX],
        color=color,
        linestyle="--",
        zorder=zorder,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    snapshots_off = load_snapshots(combo=COMBO_OFF)
    snapshots_on = load_snapshots(combo=COMBO_ON)
    representative_off = snapshots_off[-1]
    representative_on = snapshots_on[-1]
    figure, panel = manage_figure.create_figure(
        ## chosen by eye
        panel_aspect_ratio=6.0 / 5.0,
        ## drawn at the width the paper prints it at, so its text is the size it asks for
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    plot_even_odd_profiles(
        panel=panel,
        snapshot=representative_off,
        color="red",
        zorder=2,
    )
    plot_even_odd_profiles(
        panel=panel,
        snapshot=representative_on,
        color="blue",
        zorder=3,
    )
    panel.set_ylabel(r"$p$", labelpad=10.0)
    panel.set_xlabel(r"$x_0$")
    panel.set_ylim((24.4, 30.2))
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.085,
        y_pos_fraction=0.65,
        label=rf"$t = {representative_off.step_time:.2f}$",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        ## math italic reads smaller than upright text, so this sits with the axis labels
        text_size_pt=paper_style.FIGURE_PARAMS.text_size_params.axis_label_size_pt,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=["-", "--"],
        labels=["even row", "odd row"],
        colors=["black", "black"],
        anchor_point_fraction=(0.025, 0.975),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=["o", "o"],
        labels=["carbuncle phenomenon", "corrected shock-anisotropy"],
        colors=["red", "blue"],
        anchor_point_fraction=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
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
