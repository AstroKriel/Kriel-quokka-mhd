## { SCRIPT

##
## === DEPENDENCIES
##

import re
from pathlib import Path

import numpy
from jormi.ww_plots import manage_plots, style_plots

##
## === CONSTANTS
##

GAMMA = 5.0 / 3.0  # quokka::EOS_Traits<MHDQuirk>::gamma

## data figure convention: interpolation order is the only axis varied here (see paper thread:
## this test checks whether the recommended combo's extremum-preserving limiter compromises the
## carbuncle-robustness plain PPM provides, following up on the current-sheet PPM-vs-PPM-EP finding)
COLOR_MAP = {"q26-b25-ppm": "cornflowerblue", "q26-b25-ppm_ep": "orangered"}
MARKER_MAP = {"q26-b25-ppm": "o", "q26-b25-ppm_ep": "D"}
LABEL_MAP = {"q26-b25-ppm": "PPM", "q26-b25-ppm_ep": "PPM-EP"}

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/quirk/ncells=128"
FIGURE_PATH = ROOT_DIR / "figures/problems/quirk/ncells=128/peak_entropy_asymmetry.png"

##
## === HELPER FUNCTIONS
##


def compute_peak_entropy_asymmetry(
    *,
    density: numpy.ndarray,
    pressure: numpy.ndarray,
) -> float:
    """Return the largest jump in `s = p / rho^gamma` between transversely (`x_1`) adjacent cells.

    This mirrors the odd-even entropy check in `testMHDQuirk.cpp` (`computeAfterEvolve`), but is
    computed over the full `(x_0, x_1)` slice rather than at a single fixed `x_0` index, so it does
    not depend on knowing exactly where that index sits relative to the shock.
    """
    entropy = pressure / density**GAMMA
    transverse_jumps = numpy.abs(numpy.diff(entropy, axis=1))
    return float(transverse_jumps.max())


def load_combo_time_series(
    *,
    combo_dir: Path,
):
    diagnostics_dir = combo_dir / "diagnostics"
    density_paths = sorted(
        diagnostics_dir.glob("density-slice=x_2-index=*.npz"),
        key=lambda path: int(re.search(r"index=(\d+)", path.stem).group(1)),
    )
    step_times = []
    peak_asymmetries = []
    for density_path in density_paths:
        index = re.search(r"index=(\d+)", density_path.stem).group(1)
        pressure_path = diagnostics_dir / f"pressure-slice=x_2-index={index}.npz"
        density_data = numpy.load(density_path)
        pressure_data = numpy.load(pressure_path)
        step_times.append(float(density_data["step_time"]))
        peak_asymmetries.append(
            compute_peak_entropy_asymmetry(
                density=density_data["sarray_2d"],
                pressure=pressure_data["sarray_2d"],
            ),
        )
    return numpy.array(step_times), numpy.array(peak_asymmetries)


##
## === PROGRAM MAIN
##


def main():
    style_plots.set_theme()
    fig, ax = manage_plots.create_figure()

    for combo_name in ["q26-b25-ppm", "q26-b25-ppm_ep"]:
        step_times, peak_asymmetries = load_combo_time_series(combo_dir=DATASET_DIR / combo_name)
        ax.plot(
            step_times,
            peak_asymmetries,
            color=COLOR_MAP[combo_name],
            marker=MARKER_MAP[combo_name],
            markeredgecolor=COLOR_MAP[combo_name],
            markerfacecolor="none",
            markersize=5,
            linestyle="-",
            linewidth=1.0,
            label=LABEL_MAP[combo_name],
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\max \left| \Delta (p / \rho^\gamma) \right|$")
    ax.legend(loc="best")

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
