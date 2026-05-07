## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import pandas

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots

##
## === PROGRAM MAIN
##

def main() -> None:
    datasets_dir = Path(__file__).parents[2] / "datasets"
    figures_dir = Path(__file__).parents[2] / "figures" / "scalings"
    manage_io.create_directory(figures_dir)
    df = pandas.read_csv(datasets_dir / "scalings" / "weak_gpu_scaling.csv")
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
        "Q26": "cornflowerblue",
        "B25": "orangered",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "o",
        "B25": "D",
    }
    marker_size_map = {
        "LD04": 60,
        "B25": 120,
    }
    df["updates_per_s_per_gpu"] = 1e6 / (df["us_per_zone_update"] * df["num_gpus"])
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
        ax.scatter(
            x=group["num_gpus"],
            y=group["updates_per_s_per_gpu"],
            c="none",
            marker=marker_map[avg],
            s=marker_size_map[avg],
            edgecolors=color_map[compute],
            linewidths=1.5,
            label=f"{compute} + {avg}",
        )
    ax.set_xlabel("GPU count")
    ax.set_ylabel("Zone updates per second per GPU")
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
        bottom=8e6,
        top=6e7,
    )
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
