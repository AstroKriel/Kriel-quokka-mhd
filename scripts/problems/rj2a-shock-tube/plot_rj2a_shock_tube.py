## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path

import numpy
from jormi.ww_plots import manage_plots, style_plots
from ww_quokka_sims.sim_io.profile_models import ScalarProfile, VectorProfile

from riemann_solver.exact_solution import evaluate, solve_riemann_problem
from riemann_solver.mhd_state import PrimitiveState, compute_energy

##
## === CONSTANTS
##

BX = 0.5641895835477562
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
    base_dir = Path(__file__).parents[3] / "datasets/problems/rj2a-shock-tube/ncells=512/derived"
    rho_profile = ScalarProfile.load_from_file(base_dir / "density-axis=x_0-index=0000766.json")
    pressure_profile = ScalarProfile.load_from_file(base_dir / "pressure-axis=x_0-index=0000766.json")
    total_energy_profile = ScalarProfile.load_from_file(base_dir / "total_energy-axis=x_0-index=0000766.json")
    vel_profile = VectorProfile.load_from_file(base_dir / "velocity-axis=x_0-index=0000766.json")
    mag_profile = VectorProfile.load_from_file(base_dir / "magnetic-axis=x_0-index=0000766.json")
    step_time = rho_profile.step_time

    left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
    right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
    solution = solve_riemann_problem(left=left, right=right, bx=BX, gamma=GAMMA)
    exact_x = numpy.linspace(0.0, 1.0, 2001)
    exact_states = evaluate(solution=solution, x=exact_x, t=step_time, x0=X0)
    exact_rho = numpy.array([state.rho for state in exact_states])
    exact_pressure = numpy.array([state.p for state in exact_states])
    exact_vx = numpy.array([state.u for state in exact_states])
    exact_vy = numpy.array([state.v for state in exact_states])
    exact_vz = numpy.array([state.w for state in exact_states])
    exact_by = numpy.array([state.by for state in exact_states])
    exact_bz = numpy.array([state.bz for state in exact_states])
    exact_energy = numpy.array([compute_energy(state=state, bx=BX, gamma=GAMMA) for state in exact_states])

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
        fig_path=Path(__file__).parents[3] / "figures/problems/rj2a-shock-tube/rj2a-profiles.png",
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
