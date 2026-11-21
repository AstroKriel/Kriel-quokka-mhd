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
    label: str
    axis_y_range: tuple[float, float]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/quirk"
FIGURE_DIR: Path = ROOT_DIR / "figures/problems/quirk"
PAPER_FIGURE_PATH: Path = FIGURE_DIR / "odd-even-split-hydro.png"
RESPONSE_FIGURE_PATH: Path = FIGURE_DIR / "odd-even-split-comparison.png"

## the y-ranges place each corrected post-shock plateau at the same height in its panel
HYDRO_CONFIG: VariantConfig = VariantConfig(
    dataset_dir=DATASET_DIR / "hydro/num_cells=128",
    label="Hydro",
    axis_y_range=(24.4, 28.5),
)
MAGNETISED_CONFIG: VariantConfig = VariantConfig(
    dataset_dir=DATASET_DIR / "magnetised/num_cells=128",
    label="MHD",
    axis_y_range=(18.0, 21.0),
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


##
## === PANEL FUNCTIONS
##


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


def add_shock_direction_arrow(
    *,
    panel: manage_figure.Panel,
) -> None:
    figure_params = style_figure.get_figure_params()
    arrow_y_fraction = 0.68
    arrow_start_x_fraction = 0.58
    panel.annotate(
        text="",
        xy=(0.85, arrow_y_fraction),
        xytext=(arrow_start_x_fraction, arrow_y_fraction),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linewidth": figure_params.artist_params.line_width_pt,
            "mutation_scale": figure_params.text_size_params.annotation_size_pt,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    panel.annotate(
        text="shock",
        xy=(arrow_start_x_fraction, arrow_y_fraction),
        xycoords="axes fraction",
        xytext=(-6.0, 0.0),
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="center",
        color="black",
        fontsize=figure_params.text_size_params.legend_size_pt,
    )


def add_legends(
    *,
    panel: manage_figure.Panel,
) -> None:
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


def plot_odd_even_split(
    *,
    panel: manage_figure.Panel,
    variant_config: VariantConfig,
) -> None:
    snapshot_off = load_pressure_snapshot(
        extracted_dir=variant_config.dataset_dir / "q26-b25-ppm_ep-no-carbuncle-fix" / "extracted",
    )
    snapshot_on = load_pressure_snapshot(
        extracted_dir=variant_config.dataset_dir / "q26-b25-ppm_ep" / "extracted",
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
    add_shock_direction_arrow(panel=panel)
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


##
## === FIGURE FUNCTIONS
##


def plot_paper_figure() -> None:
    figure, panel = manage_figure.create_figure(
        panel_aspect_ratio=6.0 / 5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=paper_style.column_width(fraction=1.0),
        ),
    )
    plot_odd_even_split(
        panel=panel,
        variant_config=HYDRO_CONFIG,
    )
    add_legends(panel=panel)
    manage_figure.save_figure(
        figure=figure,
        figure_path=PAPER_FIGURE_PATH,
    )


def plot_response_figure() -> None:
    figure, panel_grid = manage_figure.create_figure(
        num_panel_cols=2,
        num_panel_rows=1,
        panel_aspect_ratio=6.0 / 5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=paper_style.page_width(fraction=0.85),
        ),
    )
    variant_configs = (HYDRO_CONFIG, MAGNETISED_CONFIG)
    for _panel, _variant_config in zip(panel_grid[0], variant_configs):
        plot_odd_even_split(
            panel=_panel,
            variant_config=_variant_config,
        )
        annotate_panel.add_text(
            panel=_panel,
            x_pos_fraction=0.085,
            y_pos_fraction=0.47,
            label=_variant_config.label,
            x_alignment=box_positions.Positions.Side.Left,
            y_alignment=box_positions.Positions.Side.Top,
            text_size_pt=paper_style.FIGURE_PARAMS.text_size_params.axis_label_size_pt,
        )
    add_legends(panel=panel_grid[0, 0])
    panel_grid[0, 1].tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=False,
        labelright=True,
    )
    panel_grid[0, 1].yaxis.set_label_position("right")
    manage_figure.save_figure(
        figure=figure,
        figure_path=RESPONSE_FIGURE_PATH,
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
    plot_paper_figure()
    plot_response_figure()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
