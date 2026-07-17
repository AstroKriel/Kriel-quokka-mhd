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

from aegir import exact_solution, mhd_state

##
## === CONSTANTS
##

MAGNETIC_FIELD_NORMAL = 0.5641895835477562
GAMMA = 5.0 / 3.0
X0 = 0.5  # initial discontinuity position in quokka's [0, 1] domain

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

## recommended combo, used for the LLF-vs-HLLD comparison (matched resolution+combo)
RECOMMENDED_COMBO = "q26-b25-ppm_ep"

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/ryu-jones-2a-shock-tube/ncells=512"
FIGURE_PATH = ROOT_DIR / "figures/problems/ryu-jones-2a-shock-tube/ncells=512/scheme_comparison.png"

##
## === HELPER FUNCTIONS
##


def load_combo_profiles(
    *,
    combo_dir: Path,
):
    extracted_dir = combo_dir / "extracted"
    density_path = next(extracted_dir.glob("density-axis=x_0-index=*.json"))
    index = density_path.stem.split("index=")[-1]
    return {
        "density": ScalarProfile.load_from_file(extracted_dir / f"density-axis=x_0-index={index}.json"),
        "pressure": ScalarProfile.load_from_file(extracted_dir / f"pressure-axis=x_0-index={index}.json"),
        "total_energy":
        ScalarProfile.load_from_file(extracted_dir / f"total_energy-axis=x_0-index={index}.json"),
        "velocity": VectorProfile.load_from_file(extracted_dir / f"velocity-axis=x_0-index={index}.json"),
        "magnetic": VectorProfile.load_from_file(extracted_dir / f"magnetic-axis=x_0-index={index}.json"),
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
    axs[1, 1].plot(profiles["total_energy"].position, profiles["total_energy"].field_value, **plot_args)
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
    axs[3, 0].plot(
        profiles["velocity"].components["x_2"].position,
        profiles["velocity"].components["x_2"].field_value,
        **plot_args,
    )
    axs[3, 1].plot(
        profiles["magnetic"].components["x_2"].position,
        profiles["magnetic"].components["x_2"].field_value,
        **plot_args,
    )


def plot_exact_profile(
    exact_x,
    exact_fields,
    axs,
):
    plot_args = dict(color="black", linewidth=1.5, linestyle="-", zorder=4)
    axs[0, 0].plot(exact_x, exact_fields["density"], **plot_args)
    axs[0, 1].plot(exact_x, exact_fields["pressure"], **plot_args)
    axs[1, 0].plot(exact_x, exact_fields["velocity_normal"], **plot_args)
    axs[1, 1].plot(exact_x, exact_fields["total_energy"], **plot_args)
    axs[2, 0].plot(exact_x, exact_fields["velocity_transverse_1"], **plot_args)
    axs[2, 1].plot(exact_x, exact_fields["magnetic_field_transverse_1"], **plot_args)
    axs[3, 0].plot(exact_x, exact_fields["velocity_transverse_2"], **plot_args)
    axs[3, 1].plot(exact_x, exact_fields["magnetic_field_transverse_2"], **plot_args)


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
    axs[1, 1].plot(profiles["total_energy"].position, profiles["total_energy"].field_value, **plot_args)
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
    axs[3, 0].plot(
        profiles["velocity"].components["x_2"].position,
        profiles["velocity"].components["x_2"].field_value,
        **plot_args,
    )
    axs[3, 1].plot(
        profiles["magnetic"].components["x_2"].position,
        profiles["magnetic"].components["x_2"].field_value,
        **plot_args,
    )


def compute_exact_solution():
    left_state = mhd_state.PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=0.95,
    )
    right_state = mhd_state.PrimitiveState(
        density=1.0,
        velocity_normal=0.0,
        velocity_transverse_1=0.0,
        velocity_transverse_2=0.0,
        magnetic_field_transverse_1=1.1283791670955125,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=1.0,
    )
    riemann_solution = exact_solution.solve_riemann_problem(
        left_state=left_state,
        right_state=right_state,
        magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
        gamma=GAMMA,
    )
    ## use the step_time recorded in one of the combo's extracted files so the exact solution is
    ## sampled at the same physical time the simulations reached, not a hardcoded stop_time
    combo_dir = DATASET_DIR / "hlld" / RECOMMENDED_COMBO
    step_time = ScalarProfile.load_from_file(
        next((combo_dir / "extracted").glob("density-axis=x_0-index=*.json")),
    ).step_time
    exact_x = numpy.linspace(0.0, 1.0, 2001)
    exact_states = exact_solution.sample_snapshot(
        riemann_solution=riemann_solution,
        positions=exact_x,
        time=step_time,
        discontinuity_position=X0,
    )
    exact_fields = {
        "density": numpy.array([state.density for state in exact_states]),
        "pressure": numpy.array([state.pressure for state in exact_states]),
        "velocity_normal": numpy.array([state.velocity_normal for state in exact_states]),
        "velocity_transverse_1": numpy.array([state.velocity_transverse_1 for state in exact_states]),
        "velocity_transverse_2": numpy.array([state.velocity_transverse_2 for state in exact_states]),
        "magnetic_field_transverse_1":
        numpy.array([state.magnetic_field_transverse_1 for state in exact_states]),
        "magnetic_field_transverse_2":
        numpy.array([state.magnetic_field_transverse_2 for state in exact_states]),
        "total_energy":
        numpy.array(
            [
                mhd_state.compute_total_energy(
                    state=state,
                    magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
                    gamma=GAMMA,
                ) for state in exact_states
            ],
        ),
    }
    return exact_x, exact_fields


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
        anchor_point=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
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
        anchor_point=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
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
    exact_x, exact_fields = compute_exact_solution()

    llf_profiles = load_combo_profiles(combo_dir=DATASET_DIR / "llf" / RECOMMENDED_COMBO)

    fig, axs = manage_plots.create_figure(num_cols=2, num_rows=4, share_x=True)
    plot_exact_profile(exact_x, exact_fields, axs)
    plot_llf_profile(llf_profiles, axs)
    for compute_key in EMF_COMPUTES:
        for averaging_key in EMF_AVERAGINGS:
            combo_dir = DATASET_DIR / "hlld" / f"{compute_key}-{averaging_key}-ppm_ep"
            profiles = load_combo_profiles(combo_dir=combo_dir)
            plot_combo_profile(
                profiles,
                axs,
                color=COMPUTE_COLORS[compute_key],
                marker=AVERAGING_MARKERS[averaging_key],
                zorder={"b25": 1, "fs17": 2, "q26": 3}[compute_key],
            )

    add_zoom_inset(
        ax=axs[1, 1],
        bbox=(0.4, 0.05, 0.85 - 0.4, 0.7 - 0.05),
        xlim=(0.45, 0.75),
        ylim=(4.3, 4.6),
    )
    add_zoom_inset(
        ax=axs[3, 0],
        bbox=(0.05, 0.05, 0.45 - 0.05, 0.75 - 0.05),
        xlim=(0.5, 0.75),
        ylim=(0.15, 0.35),
    )

    add_compute_legend(ax=axs[0, 0])
    add_averaging_legend(ax=axs[0, 1])
    add_llf_legend(ax=axs[1, 0])

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_0$")
    axs[1, 1].set_ylabel(r"$e_\mathrm{tot}$")
    axs[2, 0].set_ylabel(r"$u_1$")
    axs[2, 1].set_ylabel(r"$b_1$")
    axs[3, 0].set_ylabel(r"$u_2$")
    axs[3, 1].set_ylabel(r"$b_2$")
    axs[3, 0].set_xlabel(r"$x_0$")
    axs[3, 1].set_xlabel(r"$x_0$")

    for ax in [axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1]]:
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
