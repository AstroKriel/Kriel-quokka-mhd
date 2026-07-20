## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io, manage_io, manage_log
from jormi.ww_plots import manage_plots, style_plots, annotate_axis
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

## oblique slow-wave test: k = (1, 2, 3), 45 degrees between k and the background field, at a
## moderate resolution. b_0 carries the primary wave polarisation for this k / background-field
## geometry; b_2, out of the k-B0 plane, should be exactly zero by symmetry. Following the check
## Felker & Stone (2018) make of B_z for their field-loop test (their Fig. 13-14: B_z should stay
## zero to round-off as spurious terms enter the induction equation), we track whether b_2 stays
## negligible relative to the real wave amplitude, or grows into dynamical significance, as a
## probe of grid-imprinting from the non-grid-aligned reconstruction.
NCELLS = 128
SCHEME = "q26-b25-ppm_ep"
PROFILE_AXIS = "x_0"
PRIMARY_COMPONENT = "x_0"
SPURIOUS_COMPONENT = "x_2"

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/slow-wave/correctness/nx=1-ny=2-nz=3"
FIGURE_PATH = ROOT_DIR / "figures/problems/slow-wave/polarisation-leakage.png"

##
## === HELPER FUNCTIONS
##


def load_energy_time_series(
    *,
    component: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return (time, line-integrated energy) of `component`'s deviation from its mean, per snapshot.

    The domain is periodic, so this uses a plain equal-weight Riemann sum (sum * dx), not
    `numpy.trapezoid`: trapezoidal quadrature halves the weight of the first and last sample,
    which is only correct for an open (non-periodic) interval. On a periodic grid there is no such
    edge, and using trapz here introduced a spurious ~1-2% time-dependent oscillation, since the
    profile's phase shifts snapshot to snapshot -- the plain sum is stable to ~0.2% instead.
    """
    extracted_dir = DATASET_DIR / f"ncells={NCELLS}" / SCHEME / "extracted"
    file_paths = sorted(
        extracted_dir.glob(f"magnetic-axis={PROFILE_AXIS}-index=*.json"),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    times = []
    energies = []
    for file_path in file_paths:
        data = json_io.read_json_file_into_dict(file_path, verbose=False)
        position = numpy.asarray(data["field_comps"][component]["position"])
        field_value = numpy.asarray(data["field_comps"][component]["field_value"])
        deviation_squared = (field_value - field_value.mean())**2
        cell_size = position[1] - position[0]
        times.append(data["step_time"])
        energies.append(deviation_squared.sum() * cell_size)
    return numpy.asarray(times), numpy.asarray(energies)


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
    times, primary_energy = load_energy_time_series(component=PRIMARY_COMPONENT)
    _, spurious_energy = load_energy_time_series(component=SPURIOUS_COMPONENT)
    log10_leakage_ratio = numpy.log10(spurious_energy / primary_energy)
    tail = log10_leakage_ratio[len(log10_leakage_ratio) // 2:]
    tail_ave = tail.mean()
    tail_std = tail.std()
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(5, 6),
        fig_scale=0.9,
    )
    ax = axs[0, 0]
    ax.plot(
        times,
        log10_leakage_ratio,
        color="black",
        marker="o",
        markersize=8,
        linewidth=1.0,
        zorder=2,
    )
    ax.axhspan(
        tail_ave - tail_std,
        tail_ave + tail_std,
        color="black",
        alpha=0.15,
        linewidth=0.0,
        zorder=0,
    )
    ax.axhline(
        tail_ave,
        color="black",
        linestyle=":",
        linewidth=1.5,
        zorder=1,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.975,
        y_pos=0.775,
        label=rf"${tail_ave:.2f} \pm {tail_std:.2f}$",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Center.Center,
        text_size=20,
        text_color="black",
    )
    ax.set_ylim((-10, -6))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(
        r"$\log_{10}\!\left("
        r"\dfrac{\int (b_2 - \langle b_2 \rangle)^2 \mathrm{d}x_0}"
        r"{\int (b_0 - \langle b_0 \rangle)^2 \mathrm{d}x_0}"
        r"\right)$",
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
