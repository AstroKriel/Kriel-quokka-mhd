## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import manage_plots, style_plots

##
## === CONFIGURATION
##

## resistive Alfven-wave decay: b_2(x, t) = b_amp * exp(-gamma * t) * sin(k * x)
K_MODE = 2.0 * numpy.pi  ## one mode in a unit-length box

NCELLS = 256
SCHEME = "q26-b25-ppm_ep"
PROFILE_AXIS = "x_0"
DECAYING_COMPONENT = "x_2"  ## Alfven perturbation: transverse to both k and the background field

## log-spaced resistivity sweep at fixed resolution; eta=0 is excluded from the log-log comparison
## (gamma_analytic = 0 there) but is still loaded, as a check that the measured decay rate vanishes.
## dataset directories are keyed by this exact string (not a reformatted float), since e.g. "1e-05"
## does not match the on-disk directory name "0.00001"
ETA_LABELS = ("0", "0.00001", "0.0000316", "0.0001", "0.000316", "0.001", "0.00316", "0.01", "0.0316")

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/alfven-wave-linear/correctness/resistive"
FIGURE_PATH = ROOT_DIR / "figures/problems/alfven-wave-linear/decay.png"

##
## === HELPER FUNCTIONS
##


def load_component_snapshots(
    *,
    eta_label: str,
    component: str,
) -> list[tuple[float, numpy.ndarray, numpy.ndarray]]:
    """Load (time, position, field_value) for every snapshot of one magnetic-field component."""
    extracted_dir = DATASET_DIR / f"eta={eta_label}" / f"ncells={NCELLS}" / SCHEME / "extracted"
    snapshots = []
    file_paths = sorted(extracted_dir.glob(f"magnetic-axis={PROFILE_AXIS}-index=*.json"))
    for file_path in file_paths:
        data = json_io.read_json_file_into_dict(file_path, verbose=False)
        comp = data["field_comps"][component]
        snapshots.append((data["step_time"], numpy.asarray(comp["position"]), numpy.asarray(comp["field_value"])))
    return sorted(snapshots, key=lambda snapshot: snapshot[0])


def measure_decay_rate(
    *,
    eta_label: str,
) -> float:
    """Measure gamma from a linear fit of ln(amplitude) vs time across every saved snapshot.

    Using every snapshot (rather than just the first and last) averages over the oscillation in
    the instantaneous peak amplitude and is far less sensitive to noise in any single snapshot.
    """
    snapshots = load_component_snapshots(eta_label=eta_label, component=DECAYING_COMPONENT)
    times = numpy.asarray([snapshot[0] for snapshot in snapshots])
    amplitudes = numpy.asarray([numpy.max(numpy.abs(snapshot[2])) for snapshot in snapshots])
    slope, _ = numpy.polyfit(times, numpy.log(amplitudes), deg=1)
    return -slope


def plot_decay_rate_panel(
    *,
    ax,
) -> None:
    """Plot the measured decay rate vs resistivity, against the analytic gamma = eta * k^2 / 2 line."""
    nonzero_eta_labels = [label for label in ETA_LABELS if float(label) > 0.0]
    nonzero_etas = numpy.asarray([float(label) for label in nonzero_eta_labels])
    measured_gammas = numpy.asarray([
        measure_decay_rate(eta_label=label) for label in nonzero_eta_labels
    ])
    analytic_gammas = nonzero_etas * K_MODE**2 / 2.0
    ax.plot(
        numpy.log10(nonzero_etas),
        numpy.log10(analytic_gammas),
        color="black",
        linestyle=":",
        linewidth=1.5,
        zorder=1,
        label=r"$\gamma = \eta k^2 / 2$",
    )
    ax.plot(
        numpy.log10(nonzero_etas),
        numpy.log10(measured_gammas),
        color="black",
        marker="o",
        markersize=9,
        linestyle="",
        zorder=2,
    )
    ## sanity check (not plotted: log10(0) is undefined): the eta=0 run should show no measurable decay
    zero_eta_gamma = measure_decay_rate(eta_label="0")
    print(f"measured gamma at eta=0 (should be ~0): {zero_eta_gamma:.3e}")


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(5, 6),
    )
    plot_decay_rate_panel(ax=axs[0, 0])
    axs[0, 0].set_xlabel(r"$\log_{10}\ (\mathrm{input}\ \eta)$")
    axs[0, 0].set_ylabel(r"$\log_{10}\ (\mathrm{measured}\ \gamma)$")
    axs[0, 0].legend(loc="upper left", frameon=False)
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=200,
    )

##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
