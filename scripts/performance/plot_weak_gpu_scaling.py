## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy
import pandas

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots, style_plots

##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    datasets_dir = Path(__file__).parents[2] / "datasets"
    figures_dir = Path(__file__).parents[2] / "figures" / "performance"
    manage_io.create_directory(figures_dir)
    df = pandas.read_csv(datasets_dir / "performance" / "weak_gpu_scaling.csv")
    df["num_gpus"] = pandas.to_numeric(
        df["num_gpus"],
        errors="coerce",
    )
    df["us_per_zone_update"] = pandas.to_numeric(
        df["us_per_zone_update"],
        errors="coerce",
    )
    df = df.dropna(subset=["num_gpus", "us_per_zone_update", "compute_scheme", "averaging_scheme"])
    color_map = {
        "Q26": "gold",
        "B25": "cornflowerblue",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "o",
        "B25": "D",
    }
    linestyle_map = {
        "LD04": "-",
        "B25": "--",
    }
    marker_size_map = {
        "LD04": 90,
        "B25": 90,
    }
    ## draw order (top to bottom): Q26, then B25, then FS18
    zorder_map = {
        "Q26": 3,
        "B25": 2,
        "FS18": 1,
    }
    df["updates_per_s_per_gpu"] = 1.0 / (df["us_per_zone_update"] * df["num_gpus"])
    compute_order = ["Q26", "B25", "FS18"]
    avg_order = ["LD04", "B25"]
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
    df = df.sort_values(["compute_scheme", "averaging_scheme", "num_gpus"])
    fig, ax = manage_plots.create_figure()
    for (compute, avg), group in df.groupby(
            by=["compute_scheme", "averaging_scheme"],
            sort=False,
    ):
        compute = str(compute)
        avg = str(avg)
        ## perfect weak scaling: a flat line at the (averaged) 1-GPU throughput per GPU
        one_gpu_value = group.loc[group["num_gpus"] == 1, "updates_per_s_per_gpu"].mean()
        ax.axhline(
            y=one_gpu_value,
            linestyle=linestyle_map[avg],
            linewidth=1.2,
            color=color_map[compute],
            alpha=0.5,
            zorder=zorder_map[compute],
        )
        ax.scatter(
            x=group["num_gpus"],
            y=group["updates_per_s_per_gpu"],
            c=color_map[compute],
            marker=marker_map[avg],
            s=marker_size_map[avg],
            edgecolors="black",
            linewidths=1.5,
            label=f"{compute} + {avg}",
            zorder=10 + zorder_map[compute],
        )
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Mzone updates / s. / GPU")
    ax.set_xscale(
        value="log",
        base=2,
    )
    ax.set_yscale("log")
    ax.set_xlim(
        left=2**-1,
        right=2**9.5,
    )
    ax.set_ylim(
        bottom=8,
        top=6e1,
    )
    y_ticks = [10, 20, 30, 40, 50, 60]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(tick) for tick in y_ticks])
    gpu_ticks = [1, 8, 64, 512]
    ax.set_xticks(gpu_ticks)
    ax.set_xticklabels([f"$2^{{{round(numpy.log2(gpu_count))}}}$" for gpu_count in gpu_ticks])
    ax.minorticks_off()
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "weak_gpu_scaling.png",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
