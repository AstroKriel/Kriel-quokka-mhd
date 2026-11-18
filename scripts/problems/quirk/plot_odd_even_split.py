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
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.snapshots import find_snapshots

## local
from local_helpers import paper_style

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class PressureSnapshot:
    step_time: float
    pressure: numpy_typing.NDArray[numpy.floating]


@dataclass(frozen=True)
class VariantConfig:
    dataset_dir: Path
    figure_path: Path
    axis_y_range: tuple[float, float]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/quirk"
FIGURE_DIR: Path = ROOT_DIR / "figures/problems/quirk"

## the y-ranges place each corrected post-shock plateau at the same height in its panel
VARIANT_CONFIGS: tuple[VariantConfig, ...] = (
    VariantConfig(
        dataset_dir=DATASET_DIR / "hydro/num_cells=128",
        figure_path=FIGURE_DIR / "odd-even-split-hydro.png",
        axis_y_range=(24.4, 28.5),
    ),
    VariantConfig(
        dataset_dir=DATASET_DIR / "magnetised/num_cells=128",
        figure_path=FIGURE_DIR / "odd-even-split-magnetised.png",
        axis_y_range=(18.0, 21.0),
    ),
)

##
## === HELPER FUNCTIONS
##


def load_pressure_snapshot(
    *,
    extracted_dir: Path,
) -> PressureSnapshot:
    """Load the pressure slice nearest t = 0.55, before the shock reaches the x_0 = 1 boundary."""
    slice_path = find_snapshots.find_npz_near_time(
        extracted_dir=extracted_dir,
        glob_pattern="pressure-slice=x_2-index=*.npz",
        target_time=0.55,
    )
    with numpy.load(slice_path) as data:
        return PressureSnapshot(
            step_time=float(data["step_time"]),
            pressure=data["sarray_2d"],
        )


def plot_even_odd_profiles(
    *,
    panel: manage_figure.Panel,
    snapshot: PressureSnapshot,
    color: annotate_panel.ColorType,
    zorder: int,
) -> None:
    even_row_index = 0
    odd_row_index = 1
    num_cells = snapshot.pressure.shape[0]
    x_0 = (numpy.arange(num_cells) + 0.5) / num_cells
    panel.plot(
        x_0,
        snapshot.pressure[:, even_row_index],
        color=color,
        linestyle="-",
        zorder=zorder,
    )
    panel.plot(
        x_0,
        snapshot.pressure[:, odd_row_index],
        color=color,
        linestyle="--",
        zorder=zorder,
    )


def plot_odd_even_split(
    *,
    variant_config: VariantConfig,
) -> None:
    snapshot_off = load_pressure_snapshot(
        extracted_dir=variant_config.dataset_dir / "q26-b25-ppm_ep-no-carbuncle-fix" / "extracted",
    )
    snapshot_on = load_pressure_snapshot(
        extracted_dir=variant_config.dataset_dir / "q26-b25-ppm_ep" / "extracted",
    )
    figure, panel = manage_figure.create_figure(
        panel_aspect_ratio=6.0 / 5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    plot_even_odd_profiles(
        panel=panel,
        snapshot=snapshot_off,
        color="red",
        zorder=2,
    )
    plot_even_odd_profiles(
        panel=panel,
        snapshot=snapshot_on,
        color="blue",
        zorder=3,
    )
    panel.set_ylabel(r"$p$", labelpad=10.0)
    panel.set_xlabel(r"$x_0$")
    panel.set_ylim(variant_config.axis_y_range)
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.085,
        y_pos_fraction=0.35,
        label=rf"$t = {snapshot_off.step_time:.2f}$",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        text_size_pt=paper_style.FIGURE_PARAMS.text_size_params.axis_label_size_pt,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=["-", "--"],
        labels=["even row", "odd row"],
        colors=["black", "black"],
        anchor_point_fraction=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=["o", "o"],
        labels=["carbuncle phenomenon", "corrected shock-anisotropy"],
        colors=["red", "blue"],
        anchor_point_fraction=(0.025, 0.975),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )
    manage_figure.save_figure(
        figure=figure,
        figure_path=variant_config.figure_path,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    manage_io.create_directory(
        directory=FIGURE_DIR,
        verbose=False,
    )
    for variant_config in VARIANT_CONFIGS:
        plot_odd_even_split(variant_config=variant_config)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
