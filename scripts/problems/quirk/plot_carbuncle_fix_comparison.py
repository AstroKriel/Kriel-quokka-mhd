## { SCRIPT

##
## === DEPENDENCIES
##

import re
from pathlib import Path

import numpy
from jormi.ww_plots import color_palettes, manage_plots, style_plots

##
## === CONSTANTS
##

GAMMA = 5.0 / 3.0  # quokka::EOS_Traits<MHDQuirk>::gamma
COMBO_OFF = "q26-b25-ppm_ep-no-carbuncle-fix"
COMBO_ON = "q26-b25-ppm_ep"
NCELLS = 128
MAX_TIME = 0.5
NUM_SNAPSHOTS = 7

ROOT_DIR = Path(__file__).parents[3]
DATASET_ROOT = ROOT_DIR / f"datasets/problems/quirk/ncells={NCELLS}"
FIGURE_PATH = ROOT_DIR / f"figures/problems/quirk/ncells={NCELLS}/carbuncle_fix_comparison.png"

##
## === HELPER FUNCTIONS
##


def load_snapshots(combo):
    dataset_dir = DATASET_ROOT / combo
    density_paths = sorted(
        dataset_dir.glob("diagnostics/density-slice=x_2-index=*.npz"),
        key=lambda path: int(re.search(r"index=(\d+)", path.stem).group(1)),
    )
    snapshots = []
    for density_path in density_paths:
        index = density_path.stem.split("index=")[-1]
        pressure_path = dataset_dir / "diagnostics" / f"pressure-slice=x_2-index={index}.npz"
        density_data = numpy.load(density_path)
        pressure_data = numpy.load(pressure_path)
        rho = density_data["sarray_2d"]
        pres = pressure_data["sarray_2d"]
        entropy = pres / rho**GAMMA
        snapshots.append(
            {
                "step_time": float(density_data["step_time"]),
                "density_profile": rho.mean(axis=1),
                "pressure_profile": pres.mean(axis=1),
                "entropy_profile": entropy.mean(axis=1),
            },
        )
    return snapshots


def pick_evenly_spaced(snapshots, num_snapshots):
    if len(snapshots) <= num_snapshots:
        return snapshots
    indices = sorted(set(numpy.linspace(0, len(snapshots) - 1, num_snapshots).round().astype(int)))
    return [snapshots[index] for index in indices]


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()

    snapshots_off = pick_evenly_spaced(
        [snapshot for snapshot in load_snapshots(COMBO_OFF) if snapshot["step_time"] <= MAX_TIME],
        NUM_SNAPSHOTS,
    )
    snapshots_on = pick_evenly_spaced(
        [snapshot for snapshot in load_snapshots(COMBO_ON) if snapshot["step_time"] <= MAX_TIME],
        NUM_SNAPSHOTS,
    )
    palette_off = color_palettes.SequentialPalette.from_name(
        value_range=(0.0, 1.0),
        palette_name="Reds",
        palette_range=(0.15, 0.85),
    )
    palette_on = color_palettes.SequentialPalette.from_name(
        value_range=(0.0, 1.0),
        palette_name="Blues",
        palette_range=(0.15, 0.85),
    )
    colors_off = palette_off.mpl_cmap(numpy.linspace(0.0, 1.0, len(snapshots_off)))
    colors_on = palette_on.mpl_cmap(numpy.linspace(0.0, 1.0, len(snapshots_on)))

    fig, axs = manage_plots.create_figure(num_cols=1, num_rows=3, share_x=True, fig_scale=1.4)
    num_cells = len(snapshots_off[0]["density_profile"])
    x_0 = (numpy.arange(num_cells) + 0.5) / num_cells

    # within a state, draw earlier snapshots above later ones; states themselves never interleave
    # (each state's zorder spans a half-open [base, base+0.5) window)
    for snapshot, color in zip(snapshots_off, colors_off):
        line_args = dict(
            color=color,
            linewidth=1.4,
            marker="o",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=1.0 - 0.5 * (snapshot["step_time"] / MAX_TIME),
        )
        axs[0, 0].plot(x_0, snapshot["density_profile"], **line_args)
        axs[1, 0].plot(x_0, snapshot["pressure_profile"], **line_args)
        axs[2, 0].plot(x_0, snapshot["entropy_profile"], **line_args)

    for snapshot, color in zip(snapshots_on, colors_on):
        line_args = dict(
            color=color,
            linewidth=1.4,
            marker="o",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=2.0 - 0.5 * (snapshot["step_time"] / MAX_TIME),
        )
        axs[0, 0].plot(x_0, snapshot["density_profile"], **line_args)
        axs[1, 0].plot(x_0, snapshot["pressure_profile"], **line_args)
        axs[2, 0].plot(x_0, snapshot["entropy_profile"], **line_args)

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[1, 0].set_ylabel(r"$p$")
    axs[2, 0].set_ylabel(r"$p / \rho^\gamma$")
    axs[2, 0].set_xlabel(r"$x_0$")

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
