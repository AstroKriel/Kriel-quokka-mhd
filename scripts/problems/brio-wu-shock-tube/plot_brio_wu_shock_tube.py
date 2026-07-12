## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path
from matplotlib.patches import Rectangle
from jormi.ww_plots import manage_plots, style_plots
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.profile_models import ScalarProfile, VectorProfile

##
## === HELPER FUNCTIONS
##


def plot_profile(
    domain,
    values,
    ax,
    color,
    marker,
    zorder,
):
    ax.plot(
        domain,
        values,
        marker=marker,
        markeredgecolor=color,
        markerfacecolor="none",
        markersize=5,
        markeredgewidth=0.5,
        linestyle="",
        zorder=zorder,
    )


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()
    fig, axs = manage_plots.create_figure(
        num_cols=2,
        num_rows=3,
        share_x=True,
    )
    ax_inset = manage_plots.add_inset_axis(
        ax=axs[1, 1],
        bounds=(0.625, 0.35, 0.35, 0.525),
        y_label_alignment=box_positions.Positions.Side.Left,
    )
    color_map = {
        "q26": "cornflowerblue",
        "b25": "orangered",
        "fs18": "forestgreen",
    }
    marker_map = {
        "ld04": "o",
        "b25": "D",
    }
    zorder_map = {
        "q26": 1,
        "b25": 2,
        "fs18": 3,
    }
    base_dir = Path(__file__).parents[3] / "datasets/problems/brio-wu-shock-tube/ncells=512"
    for emf_scheme in ["fs18", "b25", "q26"]:
        for ave_scheme in ["ld04", "b25"]:
            scheme_name = f"{emf_scheme}-{ave_scheme}"
            data_dir = base_dir / scheme_name / "diagnostics"
            plot_args = dict(
                color=color_map[emf_scheme],
                marker=marker_map[ave_scheme],
                zorder=zorder_map[emf_scheme],
            )
            rho_profile = ScalarProfile.load_from_file(data_dir / "rho_t=0.100.json")
            pressure_profile = ScalarProfile.load_from_file(data_dir / "pressure_t=0.100.json")
            vel_profile = VectorProfile.load_from_file(data_dir / "vel_t=0.100.json")
            mag_profile = VectorProfile.load_from_file(data_dir / "mag_t=0.100.json")
            density_values = rho_profile.field_value
            pressure_values = pressure_profile.field_value
            by_values = mag_profile.components["x_1"].field_value
            plot_profile(
                domain=rho_profile.position,
                values=density_values,
                ax=axs[0, 0],
                **plot_args,
            )
            plot_profile(
                domain=pressure_profile.position,
                values=pressure_values,
                ax=axs[0, 1],
                **plot_args,
            )
            plot_profile(
                domain=rho_profile.position,
                values=pressure_values / density_values,
                ax=axs[1, 0],
                **plot_args,
            )
            plot_profile(
                domain=mag_profile.components["x_1"].position,
                values=by_values,
                ax=axs[1, 1],
                **plot_args,
            )
            ax_inset.plot(
                mag_profile.components["x_1"].position,
                by_values,
                marker=marker_map[ave_scheme],
                color=color_map[emf_scheme],
                markerfacecolor="none",
                markersize=4,
                markeredgewidth=0.75,
                linestyle="-",
                zorder=zorder_map[emf_scheme],
            )
            plot_profile(
                domain=vel_profile.components["x_0"].position,
                values=vel_profile.components["x_0"].field_value,
                ax=axs[2, 0],
                **plot_args,
            )
            plot_profile(
                domain=vel_profile.components["x_1"].position,
                values=vel_profile.components["x_1"].field_value,
                ax=axs[2, 1],
                **plot_args,
            )
    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$p / \rho$")
    axs[1, 1].set_ylabel(r"$b_2$")
    axs[2, 0].set_ylabel(r"$u_1$")
    axs[2, 1].set_ylabel(r"$u_2$")
    axs[2, 0].set_xlabel(r"$x_1$")
    axs[2, 1].set_xlabel(r"$x_1$")

    x1, x2 = 0.625, 0.85
    y1, y2 = -0.925, -0.875
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    delta_x = x2 - x1
    delta_y = y2 - y1
    ax_inset.tick_params(
        axis="both",
        which="both",
        labelsize=14,
    )
    rect = Rectangle(
        (x1, y1 - delta_y),
        delta_x,
        3 * delta_y,
        linestyle="--",
        edgecolor="black",
        facecolor="none",
        zorder=5,
    )
    axs[1, 1].add_patch(rect)

    for ax in [
            axs[0, 1],
            axs[1, 1],
            axs[2, 1],
    ]:
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
        fig_path=Path(__file__).parents[3] / "figures/problems/brio-wu-shock-tube/bw-profiles.png",
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
