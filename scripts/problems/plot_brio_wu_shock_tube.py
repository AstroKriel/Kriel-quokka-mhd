## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from matplotlib.patches import Rectangle
from jormi.ww_plots import manage_plots
from jormi.ww_types import check_positions

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


def load_and_plot_profile(
    file_path,
    ax,
    color,
    marker,
    zorder,
):
    domain, values = numpy.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1,
        unpack=True,
    )
    plot_profile(
        domain=domain,
        values=values,
        ax=ax,
        color=color,
        marker=marker,
        zorder=zorder,
    )
    return domain, values


##
## === PROGRAM MAIN
##


def main():
    fig, axs = manage_plots.create_figure(
        num_cols=2,
        num_rows=3,
        share_x=True,
    )
    ax_inset = manage_plots.add_inset_axis(
        ax=axs[1, 1],
        bounds=(0.625, 0.35, 0.35, 0.525),
        y_label_alignment=check_positions.Positions.Side.Left,
    )
    color_map = {
        "Q26": "cornflowerblue",
        "B25": "orangered",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "o",
        "B25": "D",
    }
    zorder_map = {
        "Q26": 1,
        "B25": 2,
        "FS18": 3,
    }
    base_dir = Path("/Users/necoturb/Documents/Projects/quokka/build/sims/bwst")
    for emf_scheme in ["FS18", "B25", "Q26"]:
        for ave_scheme in ["LD04", "B25"]:
            scheme_name = f"{emf_scheme}_{ave_scheme}"
            data_dir = base_dir / scheme_name
            plot_args = dict(
                color=color_map[emf_scheme],
                marker=marker_map[ave_scheme],
                zorder=zorder_map[emf_scheme],
            )
            domain, density_values = load_and_plot_profile(
                file_path=data_dir / "rho_t=0.100.csv",
                ax=axs[0, 0],
                **plot_args,
            )
            _, pressure_values = load_and_plot_profile(
                file_path=data_dir / "p_t=0.100.csv",
                ax=axs[0, 1],
                **plot_args,
            )
            _, by_values = load_and_plot_profile(
                file_path=data_dir / "mag_Y_t=0.100.csv",
                ax=axs[1, 1],
                **plot_args,
            )
            ax_inset.plot(
                domain,
                by_values,
                marker=marker_map[ave_scheme],
                color=color_map[emf_scheme],
                # markeredgecolor="black",
                markerfacecolor="none",
                markersize=4,
                markeredgewidth=0.75,
                linestyle="-",
                zorder=zorder_map[emf_scheme],
            )
            plot_profile(
                domain=domain,
                values=pressure_values / density_values,
                ax=axs[1, 0],
                **plot_args,
            )
            load_and_plot_profile(
                file_path=data_dir / "vel_X_t=0.100.csv",
                ax=axs[2, 0],
                **plot_args,
            )
            load_and_plot_profile(
                file_path=data_dir / "vel_Y_t=0.100.csv",
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
    ax_inset.tick_params(axis="both", which="both", labelsize=14)
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
        fig_path=base_dir / "bw-profiles-bg.png",
        dpi=300,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(main())

## } SCRIPT
