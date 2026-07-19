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
from numpy.typing import NDArray

## personal
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import manage_plots, style_plots

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
        label="B25",
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
        marker_size=9.5,
    )
    B25 = EMFAveragingSchemeStyle(
        label="B25",
        linestyle="--",
        marker="D",
        marker_size=9.5,
    )


@dataclass(frozen=True)
class ScalingSeries:
    emf_compute_scheme: EMFComputeScheme
    emf_averaging_scheme: EMFAveragingScheme
    num_gpus: NDArray[numpy.floating]
    updates_per_s_per_gpu: NDArray[numpy.floating]
    reference_value: float


##
## === CONSTANTS
##

## annotations
STRONG_SCALING_PROBLEM_SIZE = 512

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
                    reference_value=float(updates_per_s_per_gpu[0]),  # perfect scaling reference
                ),
            )
    return grouped_scaling_series


def plot_scaling_panel(
    *,
    ax: manage_plots.PlotAxis,
    grouped_scaling_series: list[ScalingSeries],
    add_y_label: bool,
) -> None:
    for series in grouped_scaling_series:
        emf_compute_scheme_style = series.emf_compute_scheme.value
        emf_averaging_scheme_style = series.emf_averaging_scheme.value
        ax.axhline(
            y=series.reference_value,
            linestyle=emf_averaging_scheme_style.linestyle,
            linewidth=1.2,
            color=emf_compute_scheme_style.color,
            alpha=0.5,
            zorder=emf_compute_scheme_style.zorder,
        )
        ax.plot(
            series.num_gpus,
            series.updates_per_s_per_gpu,
            linestyle="None",
            marker=emf_averaging_scheme_style.marker,
            markersize=emf_averaging_scheme_style.marker_size,
            markerfacecolor=emf_compute_scheme_style.color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"{emf_compute_scheme_style.label} + {emf_averaging_scheme_style.label}",
            zorder=10 + emf_compute_scheme_style.zorder,
        )
    ax.set_xlabel("GPUs")
    if add_y_label:
        ax.set_ylabel("Mzone updates / s. / GPU")
    ax.set_xscale(
        value="log",
        base=2,
    )
    ax.set_yscale("log")
    ax.set_ylim(
        bottom=8,
        top=6e1,
    )
    y_ticks = [10, 20, 30, 40, 50, 60]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(tick) for tick in y_ticks])
    ax.minorticks_off()


def annotate_strong_scaling_axis(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    ax.set_xlim(
        left=2**1,
        right=2**9.5,
    )
    ## show a tick at every GPU count tested, but only label the even powers of 2
    gpu_ticks = [4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(gpu_ticks)
    ax.set_xticklabels(
        [
            f"$2^{{{round(numpy.log2(gpu_count))}}}$" if round(numpy.log2(gpu_count)) % 2 == 0 else ""
            for gpu_count in gpu_ticks
        ],
    )
    ## show a tick at every GPU count tested, but only label the ones where 512/N^(1/3) lands
    ## on an exact integer cells/GPU side length
    labeled_gpu_ticks = {8, 64, 512}
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
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
    top_ax.set_xlabel("cells / GPU")


def annotate_weak_scaling_axis(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    ax.set_xlim(
        left=2**-1,
        right=2**9.5,
    )
    gpu_ticks = [1, 8, 64, 512]
    ax.set_xticks(gpu_ticks)
    ax.set_xticklabels([f"$2^{{{round(numpy.log2(gpu_count))}}}$" for gpu_count in gpu_ticks])


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    datasets_dir = Path(__file__).parents[2] / "datasets" / "performance"
    figures_dir = Path(__file__).parents[2] / "figures" / "performance"
    manage_io.create_directory(figures_dir)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=2,
        x_spacing=0.05,
        share_y=True,
    )
    strong_scaling_ax = axs[0, 0]
    grouped_strong_scaling_series = load_scaling_series(
        csv_path=datasets_dir / "strong_gpu_scaling.csv",
    )
    plot_scaling_panel(
        ax=strong_scaling_ax,
        grouped_scaling_series=grouped_strong_scaling_series,
        add_y_label=True,
    )
    annotate_strong_scaling_axis(ax=strong_scaling_ax)
    weak_scaling_ax = axs[0, 1]
    grouped_weak_scaling_series = load_scaling_series(
        csv_path=datasets_dir / "weak_gpu_scaling.csv",
    )
    plot_scaling_panel(
        ax=weak_scaling_ax,
        grouped_scaling_series=grouped_weak_scaling_series,
        add_y_label=False,
    )
    annotate_weak_scaling_axis(ax=weak_scaling_ax)
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "gpu_scaling.png",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
