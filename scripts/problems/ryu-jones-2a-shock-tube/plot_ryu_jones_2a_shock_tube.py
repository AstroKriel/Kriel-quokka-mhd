## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path

import numpy
from jormi.ww_plots import manage_plots, style_plots
from ww_quokka_sims.sim_io.profile_models import ScalarProfile, VectorProfile

from riemann_solver import exact_solution, mhd_state
from riemann_solver.mhd_state import PrimitiveState

##
## === CONSTANTS
##

MAGNETIC_FIELD_NORMAL = 0.5641895835477562
GAMMA = 5.0 / 3.0
X0 = 0.5  # initial discontinuity position in quokka's [0, 1] domain

##
## === HELPER FUNCTIONS
##


def plot_quokka_profile(
    domain,
    values,
    ax,
):
    ax.plot(
        domain,
        values,
        marker="o",
        markeredgecolor="cornflowerblue",
        markerfacecolor="none",
        markersize=4,
        markeredgewidth=0.5,
        linestyle="",
        zorder=1,
    )


def plot_exact_profile(
    domain,
    values,
    ax,
):
    ax.plot(
        domain,
        values,
        color="black",
        linewidth=1.0,
        linestyle="-",
        zorder=2,
    )


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()
    base_dir = Path(__file__).parents[3] / "datasets/problems/ryu-jones-2a-shock-tube/ncells=512/derived"
    rho_profile = ScalarProfile.load_from_file(base_dir / "density-axis=x_0-index=0000766.json")
    pressure_profile = ScalarProfile.load_from_file(base_dir / "pressure-axis=x_0-index=0000766.json")
    total_energy_profile = ScalarProfile.load_from_file(base_dir / "total_energy-axis=x_0-index=0000766.json")
    vel_profile = VectorProfile.load_from_file(base_dir / "velocity-axis=x_0-index=0000766.json")
    mag_profile = VectorProfile.load_from_file(base_dir / "magnetic-axis=x_0-index=0000766.json")
    step_time = rho_profile.step_time

    left_state = PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=0.95,
    )
    right_state = PrimitiveState(
        density=1.0,
        velocity_normal=0.0,
        velocity_transverse_1=0.0,
        velocity_transverse_2=0.0,
        magnetic_field_transverse_1=1.1283791670955125,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=1.0,
    )
    solution = exact_solution.solve_riemann_problem(
        left_state=left_state,
        right_state=right_state,
        magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
        gamma=GAMMA,
    )
    exact_x = numpy.linspace(0.0, 1.0, 2001)
    exact_states = exact_solution.sample_profile(solution=solution, x=exact_x, t=step_time, x0=X0)
    exact_rho = numpy.array([state.density for state in exact_states])
    exact_pressure = numpy.array([state.pressure for state in exact_states])
    exact_vx = numpy.array([state.velocity_normal for state in exact_states])
    exact_vy = numpy.array([state.velocity_transverse_1 for state in exact_states])
    exact_vz = numpy.array([state.velocity_transverse_2 for state in exact_states])
    exact_by = numpy.array([state.magnetic_field_transverse_1 for state in exact_states])
    exact_bz = numpy.array([state.magnetic_field_transverse_2 for state in exact_states])
    exact_energy = numpy.array(
        [mhd_state.compute_energy(state=state, magnetic_field_normal=MAGNETIC_FIELD_NORMAL, gamma=GAMMA) for state in exact_states],
    )

    fig, axs = manage_plots.create_figure(
        num_cols=2,
        num_rows=4,
        share_x=True,
    )
    plot_quokka_profile(rho_profile.position, rho_profile.field_value, axs[0, 0])
    plot_exact_profile(exact_x, exact_rho, axs[0, 0])
    plot_quokka_profile(pressure_profile.position, pressure_profile.field_value, axs[0, 1])
    plot_exact_profile(exact_x, exact_pressure, axs[0, 1])
    plot_quokka_profile(vel_profile.components["x_0"].position, vel_profile.components["x_0"].field_value, axs[1, 0])
    plot_exact_profile(exact_x, exact_vx, axs[1, 0])
    plot_quokka_profile(vel_profile.components["x_1"].position, vel_profile.components["x_1"].field_value, axs[1, 1])
    plot_exact_profile(exact_x, exact_vy, axs[1, 1])
    plot_quokka_profile(vel_profile.components["x_2"].position, vel_profile.components["x_2"].field_value, axs[2, 0])
    plot_exact_profile(exact_x, exact_vz, axs[2, 0])
    plot_quokka_profile(mag_profile.components["x_1"].position, mag_profile.components["x_1"].field_value, axs[2, 1])
    plot_exact_profile(exact_x, exact_by, axs[2, 1])
    plot_quokka_profile(mag_profile.components["x_2"].position, mag_profile.components["x_2"].field_value, axs[3, 0])
    plot_exact_profile(exact_x, exact_bz, axs[3, 0])
    plot_quokka_profile(total_energy_profile.position, total_energy_profile.field_value, axs[3, 1])
    plot_exact_profile(exact_x, exact_energy, axs[3, 1])

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_1$")
    axs[1, 1].set_ylabel(r"$u_2$")
    axs[2, 0].set_ylabel(r"$u_3$")
    axs[2, 1].set_ylabel(r"$b_2$")
    axs[3, 0].set_ylabel(r"$b_3$")
    axs[3, 1].set_ylabel(r"$E$")
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
        fig_path=Path(__file__).parents[3] / "figures/problems/ryu-jones-2a-shock-tube/ryu-jones-2a-profiles.png",
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
