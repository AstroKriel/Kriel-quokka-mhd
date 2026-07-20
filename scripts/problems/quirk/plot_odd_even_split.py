## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import annotate_axis, manage_plots, style_plots
from jormi.ww_types import box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class PressureSnapshot:
    step_time: float
    pressure: NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
COMBO_OFF = "q26-b25-ppm_ep-no-carbuncle-fix"
COMBO_ON = "q26-b25-ppm_ep"
ROOT_DIR = Path(__file__).parents[3]
DATASET_ROOT = ROOT_DIR / "datasets/problems/quirk/ncells=128"
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
    ax: manage_plots.PlotAxis,
    snapshot: PressureSnapshot,
    color: annotate_axis.ColorType,
    zorder: int,
) -> None:
    num_cells = snapshot.pressure.shape[0]
    x_0 = (numpy.arange(num_cells) + 0.5) / num_cells
    ax.plot(
        x_0,
        snapshot.pressure[:, EVEN_ROW_INDEX],
        color=color,
        linestyle="-",
        linewidth=1.5,
        zorder=zorder,
    )
    ax.plot(
        x_0,
        snapshot.pressure[:, ODD_ROW_INDEX],
        color=color,
        linestyle="--",
        linewidth=1.5,
        zorder=zorder,
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
    snapshots_off = load_snapshots(combo=COMBO_OFF)
    snapshots_on = load_snapshots(combo=COMBO_ON)
    representative_off = snapshots_off[-1]
    representative_on = snapshots_on[-1]
    fig, ax = manage_plots.create_figure(axis_shape=(5, 6))
    plot_even_odd_profiles(
        ax=ax,
        snapshot=representative_off,
        color="red",
        zorder=2,
    )
    plot_even_odd_profiles(
        ax=ax,
        snapshot=representative_on,
        color="blue",
        zorder=3,
    )
    ax.set_ylabel(r"$p$")
    ax.set_xlabel(r"$x_0$")
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.085,
        y_pos=0.665,
        label=rf"$t = {representative_off.step_time:.2f}$",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        text_size=20,
        text_color="black",
    )
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["-", "--"],
        labels=["even row", "odd row"],
        colors=["black", "black"],
        anchor_point=(0.025, 0.975),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
        text_size=20,
    )
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["o", "o"],
        labels=["carbuncle phenomenon", "corrected shock-anisotropy"],
        colors=["red", "blue"],
        anchor_point=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
        text_size=20,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=300,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
