## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path

from jormi.ww_plots import manage_plots, style_plots
from ww_quokka_sims.sim_io.profile_models import ScalarProfile, VectorProfile

##
## === CONSTANTS
##

## scheme tokens, in the order they appear in each dataset directory name:
##     <emf-compute>-<emf-averaging>-ppm_ep
EMF_COMPUTES = ("q26", "fs17", "b25")
EMF_AVERAGINGS = ("ld04", "b25")

## data figure convention (distinct from the schematic centering colours): blue/red/green denote the
## Q26/Balsara2025/FelkerStone2017 EMF reconstruction schemes; diamond/circle denote Balsara2025/
## LondrilloDelZanna2004 EMF averaging
COMPUTE_COLORS = {"q26": "goldenrod", "b25": "tab:blue", "fs17": "tab:green"}
AVERAGING_MARKERS = {"b25": "D", "ld04": "o"}

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


def plot_reference_profile(
    profiles,
    axs,
):
    plot_args = dict(color="black", linewidth=1.0, linestyle="-", zorder=4)
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
                zorder=3 if compute_key == "q26" else 1,
            )

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_1$")
    axs[1, 1].set_ylabel(r"$p / \rho$")
    axs[2, 0].set_ylabel(r"$u_2$")
    axs[2, 1].set_ylabel(r"$b_2$")
    axs[2, 0].set_xlabel(r"$x_1$")
    axs[2, 1].set_xlabel(r"$x_1$")

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
