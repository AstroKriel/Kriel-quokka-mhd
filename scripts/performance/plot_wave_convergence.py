## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy
import pandas
from matplotlib.ticker import NullLocator as mpl_NullLocator

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots

##
## === CONSTANTS
##

DOMAIN_LENGTH: float = 1.0

EMF_SCHEME_ORDER: list[str] = ["Q26", "B25", "FS18"]
AVE_SCHEME_ORDER: list[str] = ["LD04", "B25"]
INTERP_ORDERS: list[int] = [2, 3, 5]

COLOR_MAP: dict[str, str] = {
    "Q26": "cornflowerblue",
    "B25": "orangered",
    "FS18": "forestgreen",
}
MARKER_MAP: dict[str, str] = {
    "LD04": "D",
    "B25": "o",
}
MARKER_SIZE_MAP: dict[str, int] = {
    "LD04": 10,
    "B25": 5,
}
FILL_MAP: dict[str, int] = {
    "LD04": 0,
    "B25": 1,
}
LINESTYLE_MAP: dict[int, str] = {
    2: ":",
    3: "--",
    5: "-",
}

##
## === LOCAL HELPERS
##


def add_top_axis_resolution(
    ax: Any,
    *,
    domain_length: float = DOMAIN_LENGTH,
) -> None:
    top_ax = ax.twiny()
    top_ax.set_xscale("log")
    x0, x1 = ax.get_xlim()
    top_ax.set_xlim(
        left=x0,
        right=x1,
    )
    candidate_N = numpy.array(
        [4**2, 4**3, 4**4, 4**5, 4**6],
        dtype=float,
    )
    candidate_dx = domain_length / candidate_N
    mask = (candidate_dx <= max(x0, x1)) & (candidate_dx >= min(x0, x1))
    tick_dx = candidate_dx[mask]
    tick_N = candidate_N[mask].astype(int)
    top_ax.set_xticks(tick_dx)
    top_ax.set_xticklabels([f"{n}$^3$" for n in tick_N])
    top_ax.set_xlabel("resolution")
    top_ax.minorticks_off()
    top_ax.xaxis.set_minor_locator(mpl_NullLocator())
    top_ax.tick_params(
        axis="x",
        which="minor",
        top=False,
        bottom=False,
    )
    top_ax.tick_params(
        axis="x",
        which="major",
        top=True,
        bottom=False,
        direction="in",
    )


##
## === PIPELINE STAGES
##


def load_data(
    *,
    datasets_dir: Path,
) -> pandas.DataFrame:
    df = pandas.read_csv(datasets_dir / "wave_convergence.csv")
    df["dx"] = pandas.to_numeric(df["dx"], errors="coerce")
    df["error"] = pandas.to_numeric(df["error"], errors="coerce")
    df["interp_order"] = pandas.to_numeric(df["interp_order"], errors="coerce")
    df = df.dropna(subset=["dx", "error", "test", "emf_scheme", "ave_scheme", "interp_order"])
    df = df[(df["dx"] > 0) & (df["error"] > 0)]
    df["scaled_error"] = df["error"] / (df["dx"]**2)
    df["emf_scheme"] = pandas.Categorical(
        df["emf_scheme"],
        categories=EMF_SCHEME_ORDER,
        ordered=True,
    )
    df["ave_scheme"] = pandas.Categorical(
        df["ave_scheme"],
        categories=AVE_SCHEME_ORDER,
        ordered=True,
    )
    df["interp_order"] = pandas.Categorical(
        df["interp_order"],
        categories=INTERP_ORDERS,
        ordered=True,
    )
    df = df.sort_values(["test", "emf_scheme", "ave_scheme", "interp_order", "dx"])
    return df


def plot_convergence(
    ax: Any,
    *,
    df_test: pandas.DataFrame,
) -> None:
    for (emf_scheme, ave_scheme, p), group in df_test.groupby(
            by=["emf_scheme", "ave_scheme", "interp_order"],
            sort=False,
    ):
        emf_scheme = str(emf_scheme)
        ave_scheme = str(ave_scheme)
        p = int(p)
        ax.plot(
            group["dx"],
            group["scaled_error"],
            color=COLOR_MAP[emf_scheme],
            marker=MARKER_MAP[ave_scheme],
            markersize=MARKER_SIZE_MAP[ave_scheme],
            markerfacecolor=COLOR_MAP[emf_scheme] if FILL_MAP[ave_scheme] else "none",
            markeredgewidth=1.5,
            linestyle=LINESTYLE_MAP[p],
            linewidth=1.5,
        )


def style_axes(
    *,
    fig: Any,
    axes: list[Any],
    tests: list[str],
) -> None:
    fig.supxlabel(r"$\Delta x$")
    fig.supylabel(r"error scaled by $(\Delta x)^{2}$")
    for ax, test in zip(axes, tests):
        ax.set_title(test)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(
            left=1e-1,
            right=0.8e-4,
        )
        ax.set_ylim(
            bottom=0.5e-6,
            top=1e-3,
        )
        add_top_axis_resolution(ax)


##
## === PROGRAM MAIN
##


def main() -> None:
    datasets_dir = Path(__file__).parents[2] / "datasets" / "performance" / "wave-convergence"
    figures_dir = Path(__file__).parents[2] / "figures" / "performance" / "wave-convergence"
    manage_io.create_directory(figures_dir)
    df = load_data(datasets_dir=datasets_dir)
    tests = list(df["test"].unique())
    fig, axs = manage_plots.create_figure(
        num_rows=1,
        num_cols=2,
        share_x=True,
        share_y=True,
    )
    axes = [axs[0, 0], axs[0, 1]]
    for ax, test in zip(axes, tests):
        plot_convergence(
            ax,
            df_test=df[df["test"] == test],
        )
    style_axes(
        fig=fig,
        axes=axes,
        tests=tests,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "wave_convergence.png",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
