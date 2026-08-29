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
import pandas

from numpy import typing as numpy_typing

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions

## local
from local_helpers import paper_style

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class EMFComputeSchemeStyle:
    label: str
    color: str
    zorder: int


class EMFComputeScheme(Enum):
    Q26 = EMFComputeSchemeStyle(
        label="Q26",
        color="gold",
        zorder=3,
    )
    B25 = EMFComputeSchemeStyle(
        label="B25a",
        color="cornflowerblue",
        zorder=2,
    )
    FS18 = EMFComputeSchemeStyle(
        label="FS18",
        color="forestgreen",
        zorder=1,
    )


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    label: str
    marker: str
    linestyle: str
    marker_size: float


class EMFAveragingScheme(Enum):
    LD04 = EMFAveragingSchemeStyle(
        label="LD04",
        linestyle="-",
        marker="o",
        marker_size=5.0,
    )
    B25 = EMFAveragingSchemeStyle(
        label="B25b",
        linestyle="--",
        marker="D",
        marker_size=5.0,
    )


@dataclass(frozen=True)
class ScalingSeries:
    emf_compute_scheme: EMFComputeScheme
    emf_averaging_scheme: EMFAveragingScheme
    num_gpus: numpy_typing.NDArray[numpy.floating]
    updates_per_s_per_gpu: numpy_typing.NDArray[numpy.floating]
    reference_value: float


##
## === CONSTANTS
##

## annotations
STRONG_SCALING_PROBLEM_SIZE = 512
WEAK_SCALING_CELLS_PER_GPU = 128

##
## === HELPER FUNCTIONS
##


def load_scaling_series(
    *,
    csv_path: Path,
) -> list[ScalingSeries]:
    full_data_frame = pandas.read_csv(csv_path)
    grouped_scaling_series = []
    for emf_compute_scheme in EMFComputeScheme:
        for emf_averaging_scheme in EMFAveragingScheme:
            scheme_mask = (
                (full_data_frame["compute_scheme"] == emf_compute_scheme.name) &
                (full_data_frame["averaging_scheme"] == emf_averaging_scheme.name)
            )
            subset_data_frame = full_data_frame.loc[scheme_mask].sort_values(by="num_gpus")
            if subset_data_frame.empty:
                continue
            num_gpus = subset_data_frame["num_gpus"].to_numpy(dtype=numpy.float64)
            us_per_zone_update = subset_data_frame["us_per_zone_update"].to_numpy(dtype=numpy.float64)
            updates_per_s_per_gpu = 1.0 / (us_per_zone_update * num_gpus)
            grouped_scaling_series.append(
                ScalingSeries(
                    emf_compute_scheme=emf_compute_scheme,
                    emf_averaging_scheme=emf_averaging_scheme,
                    num_gpus=num_gpus,
                    updates_per_s_per_gpu=updates_per_s_per_gpu,
                    reference_value=float(updates_per_s_per_gpu[0]),
                ),
            )
    return grouped_scaling_series


def plot_scaling_panel(
    *,
    panel: manage_figure.Panel,
    grouped_scaling_series: list[ScalingSeries],
    add_y_label: bool,
) -> None:
    figure_params = style_figure.get_figure_params()
    theme_params = figure_params.theme_params
    for series in grouped_scaling_series:
        emf_compute_scheme_style = series.emf_compute_scheme.value
        emf_averaging_scheme_style = series.emf_averaging_scheme.value
        panel.axhline(
            y=series.reference_value,
            linestyle=emf_averaging_scheme_style.linestyle,
            linewidth=0.9,
            color=emf_compute_scheme_style.color,
            alpha=0.5,
            zorder=emf_compute_scheme_style.zorder,
        )
        panel.plot(
            series.num_gpus,
            series.updates_per_s_per_gpu,
            linestyle="None",
            marker=emf_averaging_scheme_style.marker,
            markersize=emf_averaging_scheme_style.marker_size,
            markerfacecolor=emf_compute_scheme_style.color,
            markeredgecolor=theme_params.foreground_color,
            markeredgewidth=0.6,
            label=f"{emf_compute_scheme_style.label} + {emf_averaging_scheme_style.label}",
            zorder=10 + emf_compute_scheme_style.zorder,
        )
    panel.set_xlabel("GPUs")
    if add_y_label:
        panel.set_ylabel("Mzone updates / s. / GPU")
    panel.set_xscale(
        value="log",
        base=2,
    )
    panel.set_yscale("log")
    panel.set_ylim(
        bottom=8,
        top=6e1,
    )
    y_ticks = [10, 20, 30, 40, 50, 60]
    panel.set_yticks(y_ticks)
    panel.set_yticklabels([str(tick) for tick in y_ticks])


def annotate_strong_scaling_axis(
    *,
    panel: manage_figure.Panel,
) -> None:
    panel.set_xlim(
        left=2**1,
        right=2**9.5,
    )
    gpu_ticks = [4, 8, 16, 32, 64, 128, 256, 512]
    panel.set_xticks(gpu_ticks)
    panel.set_xticklabels(
        [
            f"$2^{{{round(numpy.log2(gpu_count))}}}$" if round(numpy.log2(gpu_count)) % 2 == 0 else ""
            for gpu_count in gpu_ticks
        ],
    )
    labeled_gpu_ticks = {8, 64, 512}
    top_ax = panel.twiny()
    top_ax.set_xlim(panel.get_xlim())
    top_ax.set_xscale(
        value="log",
        base=2,
    )
    top_ax.set_xticks(gpu_ticks)
    top_ax.set_xticklabels(
        [
            f"${round(STRONG_SCALING_PROBLEM_SIZE / gpu_count ** (1.0 / 3.0))}^3$"
            if gpu_count in labeled_gpu_ticks else "" for gpu_count in gpu_ticks
        ],
    )
    top_ax.minorticks_off()
    top_ax.set_xlabel("cells / GPU", labelpad=10.0)


def annotate_weak_scaling_axis(
    *,
    panel: manage_figure.Panel,
) -> None:
    panel.set_xlim(
        left=2**(-1),
        right=2**(9.5),
    )
    gpu_ticks = [1, 8, 64, 512]
    panel.set_xticks(gpu_ticks)
    panel.set_xticklabels([f"$2^{{{round(numpy.log2(gpu_count))}}}$" for gpu_count in gpu_ticks])
    top_ax = panel.twiny()
    top_ax.set_xlim(panel.get_xlim())
    top_ax.set_xscale(
        value="log",
        base=2,
    )
    top_ax.set_xticks(gpu_ticks)
    top_ax.set_xticklabels(
        [f"${round(WEAK_SCALING_CELLS_PER_GPU * gpu_count ** (1.0 / 3.0))}^3$" for gpu_count in gpu_ticks],
    )
    top_ax.minorticks_off()
    top_ax.set_xlabel("resolution", labelpad=10.0)


def add_emf_compute_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        marker_first=False,  # put the (invisible) handle after the text, so text hugs the left edge
        anchor_point_fraction=(0.0125, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


def add_emf_averaging_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[scheme.value.linestyle for scheme in EMFAveragingScheme],
        labels=["" for _ in EMFAveragingScheme],
        colors=["black" for _ in EMFAveragingScheme],
        anchor_point_fraction=(0.0125, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[scheme.value.marker for scheme in EMFAveragingScheme],
        labels=[scheme.value.label for scheme in EMFAveragingScheme],
        colors=["black" for _ in EMFAveragingScheme],
        anchor_point_fraction=(0.0125, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    datasets_dir = Path(__file__).parents[2] / "datasets" / "performance"
    figures_dir = Path(__file__).parents[2] / "figures" / "performance"
    manage_io.create_directory(figures_dir)
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=1,
        num_panel_cols=2,
        panel_aspect_ratio=8.0 / 5.0,
        panel_col_gap_pt=5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.90),
        ),
        share_y_axis=True,
    )
    strong_scaling_ax = panel_grid[0, 0]
    grouped_strong_scaling_series = load_scaling_series(
        csv_path=datasets_dir / "strong_gpu_scaling.csv",
    )
    plot_scaling_panel(
        panel=strong_scaling_ax,
        grouped_scaling_series=grouped_strong_scaling_series,
        add_y_label=True,
    )
    annotate_strong_scaling_axis(panel=strong_scaling_ax)
    add_emf_compute_scheme_legend(panel=strong_scaling_ax)
    weak_scaling_ax = panel_grid[0, 1]
    grouped_weak_scaling_series = load_scaling_series(
        csv_path=datasets_dir / "weak_gpu_scaling.csv",
    )
    plot_scaling_panel(
        panel=weak_scaling_ax,
        grouped_scaling_series=grouped_weak_scaling_series,
        add_y_label=False,
    )
    annotate_weak_scaling_axis(panel=weak_scaling_ax)
    add_emf_averaging_scheme_legend(panel=weak_scaling_ax)
    strong_scaling_ax.set_ylim([8, 70])
    annotate_panel.add_text(
        panel=strong_scaling_ax,
        x_pos_fraction=0.95,
        y_pos_fraction=0.95,
        label="strong scaling",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Top,
    )
    annotate_panel.add_text(
        panel=weak_scaling_ax,
        x_pos_fraction=0.95,
        y_pos_fraction=0.95,
        label="weak scaling",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Top,
    )
    manage_figure.save_figure(
        figure=figure,
        figure_path=figures_dir / "gpu_scaling.png",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
