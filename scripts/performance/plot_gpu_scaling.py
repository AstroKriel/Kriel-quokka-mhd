## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
import numpy
import pandas

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots, style_plots

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ComputeSchemeStyle:
    label: str
    color: str
    zorder: int


class ComputeScheme(Enum):
    Q26 = ComputeSchemeStyle(
        label="Q26",
        color="gold",
        zorder=3,
    )
    B25 = ComputeSchemeStyle(
        label="B25",
        color="cornflowerblue",
        zorder=2,
    )
    FS18 = ComputeSchemeStyle(
        label="FS18",
        color="forestgreen",
        zorder=1,
    )


@dataclass(frozen=True)
class AveragingSchemeStyle:
    label: str
    marker: str
    linestyle: str
    marker_size: float


class AveragingScheme(Enum):
    LD04 = AveragingSchemeStyle(
        label="LD04",
        linestyle="-",
        marker="o",
        marker_size=9.5,
    )
    B25 = AveragingSchemeStyle(
        label="B25",
        linestyle="--",
        marker="D",
        marker_size=9.5,
    )


##
## === CONSTANTS
##

## strong scaling uses a fixed total problem size of 512^3 cells (see paper sec:performance:speed)
TOTAL_CELLS_PER_SIDE = 512

##
## === HELPER FUNCTIONS
##


def load_scaling_data(
    csv_path: Path,
) -> pandas.DataFrame:
    df = pandas.read_csv(csv_path)
    df["num_gpus"] = pandas.to_numeric(
        df["num_gpus"],
        errors="coerce",
    )
    df["us_per_zone_update"] = pandas.to_numeric(
        df["us_per_zone_update"],
        errors="coerce",
    )
    df = df.dropna(subset=["num_gpus", "us_per_zone_update", "compute_scheme", "averaging_scheme"])
    df["updates_per_s_per_gpu"] = 1.0 / (df["us_per_zone_update"] * df["num_gpus"])
    compute_order = [scheme.name for scheme in ComputeScheme]
    avg_order = [scheme.name for scheme in AveragingScheme]
    df["compute_scheme"] = pandas.Categorical(
        df["compute_scheme"],
        categories=compute_order,
        ordered=True,
    )
    df["averaging_scheme"] = pandas.Categorical(
        df["averaging_scheme"],
        categories=avg_order,
        ordered=True,
    )
    return df.sort_values(["compute_scheme", "averaging_scheme", "num_gpus"])


def plot_scaling_panel(
    *,
    ax: manage_plots.PlotAxis,
    df: pandas.DataFrame,
    reference_value: Callable[[pandas.DataFrame], float],
    add_y_label: bool,
) -> None:
    for group_key, group in df.groupby(
            by=["compute_scheme", "averaging_scheme"],
            sort=False,
    ):
        compute_name, avg_name = group_key  # pyright: ignore[reportGeneralTypeIssues]
        compute_style = ComputeScheme[str(compute_name)].value
        avg_style = AveragingScheme[str(avg_name)].value
        ax.axhline(
            y=reference_value(group),
            linestyle=avg_style.linestyle,
            linewidth=1.2,
            color=compute_style.color,
            alpha=0.5,
            zorder=compute_style.zorder,
        )
        ax.plot(
            group["num_gpus"],
            group["updates_per_s_per_gpu"],
            linestyle="None",
            marker=avg_style.marker,
            markersize=avg_style.marker_size,
            markerfacecolor=compute_style.color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"{compute_style.label} + {avg_style.label}",
            zorder=10 + compute_style.zorder,
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


def configure_strong_axis(
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
            f"${round(TOTAL_CELLS_PER_SIDE / gpu_count ** (1.0 / 3.0))}^3$"
            if gpu_count in labeled_gpu_ticks else "" for gpu_count in gpu_ticks
        ],
    )
    top_ax.minorticks_off()
    top_ax.set_xlabel("cells / GPU")


def configure_weak_axis(
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
    style_plots.set_theme()
    datasets_dir = Path(__file__).parents[2] / "datasets" / "performance"
    figures_dir = Path(__file__).parents[2] / "figures" / "performance"
    manage_io.create_directory(figures_dir)

    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=2,
        x_spacing=0.15,
        share_y=True,
    )
    strong_ax = axs[0, 0]
    weak_ax = axs[0, 1]

    strong_df = load_scaling_data(datasets_dir / "strong_gpu_scaling.csv")
    plot_scaling_panel(
        ax=strong_ax,
        df=strong_df,
        reference_value=lambda group: float(group.sort_values("num_gpus")["updates_per_s_per_gpu"].iloc[0]),
        add_y_label=True,
    )
    configure_strong_axis(ax=strong_ax)

    weak_df = load_scaling_data(datasets_dir / "weak_gpu_scaling.csv")
    plot_scaling_panel(
        ax=weak_ax,
        df=weak_df,
        reference_value=lambda group:
        float(group.loc[group["num_gpus"] == 1, "updates_per_s_per_gpu"].mean()),
        add_y_label=False,
    )
    configure_weak_axis(ax=weak_ax)

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
