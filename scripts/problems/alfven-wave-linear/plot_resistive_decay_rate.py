## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_data import fit_series
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_io import json_io, manage_io, manage_log
from jormi.ww_plots import manage_plots, style_plots

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class Snapshot:
    time: float
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/alfven-wave-linear/correctness/resistive"
FIGURE_PATH = ROOT_DIR / "figures/problems/alfven-wave-linear/resistive-decay-rate.png"

##
## === HELPER FUNCTIONS
##


def discover_eta_labels(
    *,
    dataset_dir: Path,
) -> tuple[str, ...]:
    """Return every `eta=<label>` directory under `dataset_dir`, sorted by resistivity value."""
    eta_dirs = sorted(
        dataset_dir.glob("eta=*"),
        key=lambda path: float(path.name.removeprefix("eta=")),
    )
    return tuple(path.name.removeprefix("eta=") for path in eta_dirs)


def load_perturbed_component_snapshots(
    *,
    eta_label: str,
) -> list[Snapshot]:
    extracted_dir = DATASET_DIR / f"eta={eta_label}" / "ncells=256" / "q26-b25-ppm_ep" / "extracted"
    snapshots = []
    file_paths = sorted(extracted_dir.glob("magnetic-axis=x_0-index=*.json"))
    for file_path in file_paths:
        dataset = json_io.read_json_file_into_dict(file_path, verbose=False)
        ## x_2: transverse to both k and the background field
        perturbed_field_comp = dataset["field_comps"]["x_2"]
        snapshots.append(
            Snapshot(
                time=dataset["step_time"],
                position=numpy.asarray(perturbed_field_comp["position"]),
                field_value=numpy.asarray(perturbed_field_comp["field_value"]),
            ),
        )
    return sorted(snapshots, key=lambda snapshot: snapshot.time)


def measure_decay_rate(
    *,
    eta_label: str,
) -> float:
    snapshots = load_perturbed_component_snapshots(
        eta_label=eta_label,
    )
    times = numpy.asarray([snapshot.time for snapshot in snapshots])
    amplitudes = numpy.asarray([numpy.max(numpy.abs(snapshot.field_value)) for snapshot in snapshots])
    fit_summary = fit_series.fit_linear_model(
        gaussian_series=GaussianSeries(
            x_values=times,
            y_values=numpy.log(amplitudes),
        ),
    )
    return -fit_summary.slope.value


def plot_decay_rate_panel(
    *,
    ax: manage_plots.PlotAxis,
    eta_labels: tuple[str, ...],
) -> None:
    input_etas = numpy.asarray([float(label) for label in eta_labels])
    measured_gammas = numpy.asarray([measure_decay_rate(eta_label=label) for label in eta_labels])
    analytic_gammas = input_etas * (2.0 * numpy.pi)**2 / 2.0
    ax.plot(
        numpy.log10(input_etas),
        numpy.log10(analytic_gammas),
        color="black",
        linestyle=":",
        linewidth=1.5,
        zorder=1,
        label=r"$\gamma = \eta k^2 / 2$",
    )
    ax.plot(
        numpy.log10(input_etas),
        numpy.log10(measured_gammas),
        color="black",
        marker="o",
        markersize=9,
        linestyle="",
        zorder=2,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    eta_labels = discover_eta_labels(dataset_dir=DATASET_DIR)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(5, 6),
    )
    plot_decay_rate_panel(
        ax=axs[0, 0],
        eta_labels=eta_labels,
    )
    axs[0, 0].set_xlabel(r"$\log_{10}\ (\mathrm{input}\ \eta)$")
    axs[0, 0].set_ylabel(r"$\log_{10}\ (\mathrm{measured}\ \gamma)$")
    axs[0, 0].legend(
        loc="upper left",
        frameon=False,
        fontsize=24,
    )
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
