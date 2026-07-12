## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path

import numpy
from jormi.ww_plots import manage_plots, style_plots
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
EMF_COMPUTES = ("q26", "fs17", "b25")
EMF_AVERAGINGS = ("ld04", "b25")

## data figure convention (distinct from the schematic centering colours): blue/red/green denote the
## Q26/Balsara2025/FelkerStone2017 EMF reconstruction schemes; diamond/circle denote Balsara2025/
## LondrilloDelZanna2004 EMF averaging
COMPUTE_COLORS = {"q26": "goldenrod", "b25": "tab:blue", "fs17": "tab:green"}
AVERAGING_MARKERS = {"b25": "D", "ld04": "o"}

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
    diagnostics_dir = combo_dir / "diagnostics"
    density_path = next(diagnostics_dir.glob("density-axis=x_0-index=*.json"))
    index = density_path.stem.split("index=")[-1]
    return {
        "density": ScalarProfile.load_from_file(diagnostics_dir / f"density-axis=x_0-index={index}.json"),
        "pressure": ScalarProfile.load_from_file(diagnostics_dir / f"pressure-axis=x_0-index={index}.json"),
        "total_energy":
        ScalarProfile.load_from_file(diagnostics_dir / f"total_energy-axis=x_0-index={index}.json"),
        "velocity": VectorProfile.load_from_file(diagnostics_dir / f"velocity-axis=x_0-index={index}.json"),
        "magnetic": VectorProfile.load_from_file(diagnostics_dir / f"magnetic-axis=x_0-index={index}.json"),
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
        markersize=4,
        markeredgewidth=0.7,
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
    plot_args = dict(color="black", linewidth=1.0, linestyle="-", zorder=4)
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
        markersize=4,
        markeredgewidth=0.7,
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
    ## use the step_time recorded in one of the combo's diagnostics so the exact solution is
    ## sampled at the same physical time the simulations reached, not a hardcoded stop_time
    combo_dir = DATASET_DIR / "hlld" / RECOMMENDED_COMBO
    step_time = ScalarProfile.load_from_file(
        next((combo_dir / "diagnostics").glob("density-axis=x_0-index=*.json")),
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
                zorder=3 if compute_key == "q26" else 1,
            )

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_1$")
    axs[1, 1].set_ylabel(r"$E$")
    axs[2, 0].set_ylabel(r"$u_2$")
    axs[2, 1].set_ylabel(r"$b_2$")
    axs[3, 0].set_ylabel(r"$u_3$")
    axs[3, 1].set_ylabel(r"$b_3$")
    axs[3, 0].set_xlabel(r"$x_1$")
    axs[3, 1].set_xlabel(r"$x_1$")

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
