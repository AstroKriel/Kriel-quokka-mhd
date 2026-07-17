## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path

import numpy
from matplotlib.lines import Line2D as mpl_line2d

from jormi.ww_plots import annotate_axis, manage_plots, style_plots
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.profile_models import ScalarProfile, VectorProfile

##
## === CONSTANTS
##

## scheme tokens, in the order they appear in each dataset directory name:
##     <emf-compute>-<emf-averaging>-ppm_ep
EMF_COMPUTES = ("q26", "b25", "fs17")
EMF_AVERAGINGS = ("b25", "ld04")

## data figure convention (distinct from the schematic centering colours): blue/red/green denote the
## Q26/Balsara2025/FelkerStone2017 EMF reconstruction schemes; diamond/circle denote Balsara2025/
## LondrilloDelZanna2004 EMF averaging
COMPUTE_COLORS = {"q26": "gold", "b25": "cornflowerblue", "fs17": "forestgreen"}
AVERAGING_MARKERS = {"b25": "D", "ld04": "o"}

## display shorthand for legend labels, distinct from the lowercase dataset-directory tokens
COMPUTE_LABELS = {"q26": "Q26", "fs17": "FS17", "b25": "B25"}
AVERAGING_LABELS = {"ld04": "LD04", "b25": "B25"}

## the high-resolution PPM-EP reference: aegir's exact solver fails on Brio-Wu's coplanar IC
## (singular rotation-discontinuity root-find), so this run stands in as the reference solution
REFERENCE_COMBO_DIR_NAME = "ncells=8192/hlld/q26-b25-ppm_ep"

## recommended combo, used for the LLF-vs-HLLD comparison (matched resolution+combo)
RECOMMENDED_COMBO = "q26-b25-ppm_ep"

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/brio-wu-shock-tube"
FIGURE_PATH = ROOT_DIR / "figures/problems/brio-wu-shock-tube/ncells=256/scheme_comparison.png"

##
## === HELPER FUNCTIONS
##


def load_profiles(
    *,
    combo_dir: Path,
):
    diagnostics_dir = combo_dir / "diagnostics"
    density_path = next(diagnostics_dir.glob("density-axis=x_0-index=*.json"))
    index = density_path.stem.split("index=")[-1]
    density = ScalarProfile.load_from_file(diagnostics_dir / f"density-axis=x_0-index={index}.json")
    pressure = ScalarProfile.load_from_file(diagnostics_dir / f"pressure-axis=x_0-index={index}.json")
    velocity = VectorProfile.load_from_file(diagnostics_dir / f"velocity-axis=x_0-index={index}.json")
    magnetic = VectorProfile.load_from_file(diagnostics_dir / f"magnetic-axis=x_0-index={index}.json")
    return {
        "density": density,
        "pressure": pressure,
        "pressure_ratio": pressure.field_value / density.field_value,
        "velocity": velocity,
        "magnetic": magnetic,
    }


def plot_combo_profile(
    profiles,
    axs,
    *,
    color: str,
    marker: str,
    zorder: int,
):
    plot_args = dict(
        marker=marker,
        markeredgecolor=color,
        markerfacecolor="none",
        markersize=6,
        markeredgewidth=0.3,
        linestyle="",
        zorder=zorder,
    )
    axs[0, 0].plot(profiles["density"].position, profiles["density"].field_value, **plot_args)
    axs[0, 1].plot(profiles["pressure"].position, profiles["pressure"].field_value, **plot_args)
    axs[1, 0].plot(
        profiles["velocity"].components["x_0"].position,
        profiles["velocity"].components["x_0"].field_value,
        **plot_args,
    )
    axs[1, 1].plot(profiles["pressure"].position, profiles["pressure_ratio"], **plot_args)
    axs[2, 0].plot(
        profiles["velocity"].components["x_1"].position,
        profiles["velocity"].components["x_1"].field_value,
        **plot_args,
    )
    axs[2, 1].plot(
        profiles["magnetic"].components["x_1"].position,
        profiles["magnetic"].components["x_1"].field_value,
        **plot_args,
    )


def smooth(
    values,
    *,
    window: int = 35,
):
    """Short boxcar moving average.

    The 8192-cell reference run stands in for an exact solution, but it's still a numerical solution
    using Q26 -- the reconstruction scheme that rings the most -- so it retains small-scale ringing of
    its own at this resolution, which does not go away (and plausibly grows) with further refinement.
    This averages over a window short enough to leave the discontinuities themselves sharp, only
    smoothing the sub-window-scale oscillations.
    """
    kernel = numpy.ones(window) / window
    pad_width = window // 2
    padded_values = numpy.pad(values, pad_width, mode="edge")
    return numpy.convolve(padded_values, kernel, mode="same")[pad_width:pad_width + len(values)]


def plot_reference_profile(
    profiles,
    axs,
):
    plot_args = dict(color="black", linewidth=1.5, linestyle="-", zorder=4)
    axs[0, 0].plot(profiles["density"].position, smooth(profiles["density"].field_value), **plot_args)
    axs[0, 1].plot(profiles["pressure"].position, smooth(profiles["pressure"].field_value), **plot_args)
    axs[1, 0].plot(
        profiles["velocity"].components["x_0"].position,
        smooth(profiles["velocity"].components["x_0"].field_value),
        **plot_args,
    )
    axs[1, 1].plot(profiles["pressure"].position, smooth(profiles["pressure_ratio"]), **plot_args)
    axs[2, 0].plot(
        profiles["velocity"].components["x_1"].position,
        smooth(profiles["velocity"].components["x_1"].field_value),
        **plot_args,
    )
    axs[2, 1].plot(
        profiles["magnetic"].components["x_1"].position,
        smooth(profiles["magnetic"].components["x_1"].field_value),
        **plot_args,
    )


def plot_llf_profile(
    profiles,
    axs,
):
    plot_args = dict(
        marker="s",
        markeredgecolor="deeppink",
        markerfacecolor="none",
        markersize=6,
        markeredgewidth=0.3,
        linestyle="",
        zorder=0,
    )
    axs[0, 0].plot(profiles["density"].position, profiles["density"].field_value, **plot_args)
    axs[0, 1].plot(profiles["pressure"].position, profiles["pressure"].field_value, **plot_args)
    axs[1, 0].plot(
        profiles["velocity"].components["x_0"].position,
        profiles["velocity"].components["x_0"].field_value,
        **plot_args,
    )
    axs[1, 1].plot(profiles["pressure"].position, profiles["pressure_ratio"], **plot_args)
    axs[2, 0].plot(
        profiles["velocity"].components["x_1"].position,
        profiles["velocity"].components["x_1"].field_value,
        **plot_args,
    )
    axs[2, 1].plot(
        profiles["magnetic"].components["x_1"].position,
        profiles["magnetic"].components["x_1"].field_value,
        **plot_args,
    )


def add_zoom_inset(
    *,
    ax,
    bbox: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Add a zoomed inset of `ax`'s data."""
    inset_ax = ax.inset_axes(bbox)
    for line in ax.get_lines():
        inset_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            marker=line.get_marker(),
            markeredgecolor=line.get_markeredgecolor(),
            markerfacecolor=line.get_markerfacecolor(),
            markersize=line.get_markersize(),
            markeredgewidth=line.get_markeredgewidth() * 3.0,
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            zorder=line.get_zorder(),
        )
    inset_ax.set_xlim(xlim)
    inset_ax.set_ylim(ylim)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    ax.indicate_inset_zoom(inset_ax, edgecolor="black")


def add_compute_legend(
    *,
    ax,
) -> None:
    """Legend for EMF compute, shown as coloured shorthand text (no marker, since colour alone
    already encodes compute in the data).
    """
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["o" for _ in EMF_COMPUTES],
        labels=[COMPUTE_LABELS[compute_key] for compute_key in EMF_COMPUTES],
        colors=[COMPUTE_COLORS[compute_key] for compute_key in EMF_COMPUTES],
        marker_size=0,  ## hide the marker handle, leaving only the coloured label text
        text_color="markerfacecolor",
        marker_first=False,  ## put the (invisible) handle after the text, so text hugs the left edge
        anchor_point=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


def add_averaging_legend(
    *,
    ax,
) -> None:
    """Legend for EMF averaging, shown as marker + shorthand."""
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[AVERAGING_MARKERS[averaging_key] for averaging_key in EMF_AVERAGINGS],
        labels=[AVERAGING_LABELS[averaging_key] for averaging_key in EMF_AVERAGINGS],
        colors=["black" for _ in EMF_AVERAGINGS],
        marker_size=7,
        text_color="black",
        anchor_point=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


def add_llf_legend(
    *,
    ax,
) -> None:
    """Legend for the LLF validation run, shown as a marker matching the data style."""
    handle = mpl_line2d(
        [0], [0],
        marker="s",
        linewidth=0,
        markeredgecolor="deeppink",
        markerfacecolor="none",
        markeredgewidth=0.3,
        markersize=7,
    )
    legend = ax.legend(
        handles=[handle],
        labels=["LLF"],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=16,
        labelcolor="black",
        frameon=False,
        borderpad=0.45,
        handletextpad=0.5,
    )
    ax.add_artist(legend)


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()
    reference_profiles = load_profiles(combo_dir=DATASET_DIR / REFERENCE_COMBO_DIR_NAME)
    llf_profiles = load_profiles(combo_dir=DATASET_DIR / "ncells=256/llf" / RECOMMENDED_COMBO)

    fig, axs = manage_plots.create_figure(num_cols=2, num_rows=3, share_x=True)
    plot_reference_profile(reference_profiles, axs)
    plot_llf_profile(llf_profiles, axs)
    for compute_key in EMF_COMPUTES:
        for averaging_key in EMF_AVERAGINGS:
            combo_dir = DATASET_DIR / "ncells=256/hlld" / f"{compute_key}-{averaging_key}-ppm_ep"
            profiles = load_profiles(combo_dir=combo_dir)
            plot_combo_profile(
                profiles,
                axs,
                color=COMPUTE_COLORS[compute_key],
                marker=AVERAGING_MARKERS[averaging_key],
                zorder={"b25": 1, "fs17": 2, "q26": 3}[compute_key],
            )

    add_zoom_inset(
        ax=axs[1, 0],
        bbox=(0.675, 0.425, 0.975 - 0.675, 0.965 - 0.425),
        xlim=(0.625, 0.85),
        ylim=(-0.31, -0.19),
    )
    add_zoom_inset(
        ax=axs[1, 1],
        bbox=(0.05, 0.3, 0.5 - 0.05, 0.925 - 0.3),
        xlim=(0.55, 0.65),
        ylim=(1.4, 1.55),
    )

    add_compute_legend(ax=axs[0, 0])
    add_averaging_legend(ax=axs[0, 1])
    add_llf_legend(ax=axs[1, 0])

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_0$")
    axs[1, 1].set_ylabel(r"$p / \rho$")
    axs[2, 0].set_ylabel(r"$u_1$")
    axs[2, 1].set_ylabel(r"$b_1$")
    axs[2, 0].set_xlabel(r"$x_0$")
    axs[2, 1].set_xlabel(r"$x_0$")

    for ax in [axs[0, 1], axs[1, 1], axs[2, 1]]:
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
        )
        ax.yaxis.set_label_position("right")

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
