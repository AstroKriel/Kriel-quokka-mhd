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
from jormi.ww_plots import annotate_axis, manage_plots, style_plots

##
## === CONFIGURATION
##

## TODO: confirm against the actual run once it lands -- placeholder values chosen to match
## the oblique linear-wave convention in `sec:tests:misaligned-waves` (k~={1,2,3}, theta=pi/4),
## not yet a real dataset. Update SCHEME/NCELLS/DATASET_DIR to match whatever the run produces.
SCHEME = "q26-b25-ppm_ep"
NCELLS = 512
NUM_PERIODS = 2
PROFILE_AXIS = "x_0"  ## any pencil direction works: it samples every phase of the oblique wave once
FIELD_NAME = "magnetic"

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / f"datasets/problems/alfven-wave-circular/correctness/angle=45-nx=1-ny=2-nz=3/{SCHEME}"
FIGURE_PATH = ROOT_DIR / "figures/problems/alfven-wave-circular/hodogram.png"

##
## === HELPER FUNCTIONS
##


def load_snapshot(
    *,
    step_index: int,
) -> tuple[float, numpy.ndarray]:
    """Load the (n_position, 3) array of field vectors for one snapshot, ordered x_0/x_1/x_2."""
    diagnostics_dir = DATASET_DIR / "diagnostics"
    file_path = diagnostics_dir / f"{FIELD_NAME}-axis={PROFILE_AXIS}-index={step_index:07d}.json"
    data = json_io.read_json_file_into_dict(file_path, verbose=False)
    components = [numpy.asarray(data["field_comps"][axis]["field_value"]) for axis in ("x_0", "x_1", "x_2")]
    return float(data["step_time"]), numpy.stack(components, axis=-1)


def transverse_basis(
    *,
    perturbation: numpy.ndarray,
) -> numpy.ndarray:
    """
    Recover the 2D plane spanned by a circularly-polarised perturbation via its top two
    singular vectors, rather than assuming a hard-coded wave-vector/background-field
    convention -- a circularly-polarised Alfven wave's perturbation vector lies in a fixed
    plane at every phase, regardless of how oblique the wave is to the grid, so this recovers
    the correct hodogram basis directly from the data.
    """
    _, _, basis = numpy.linalg.svd(perturbation, full_matrices=False)
    return basis[:2]  ## (2, 3): the two directions spanning the perturbation plane


def hodogram_coords(
    *,
    field: numpy.ndarray,
    background: numpy.ndarray,
    basis: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Project perturbation vectors onto a 2D basis to get the two hodogram coordinates."""
    perturbation = field - background
    return perturbation @ basis[0], perturbation @ basis[1]


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )

    time_initial, field_initial = load_snapshot(step_index=0)
    background = field_initial.mean(axis=0)  ## averaging over a full pencil of phases cancels the perturbation

    basis = transverse_basis(perturbation=field_initial - background)
    coord_1_initial, coord_2_initial = hodogram_coords(field=field_initial, background=background, basis=basis)

    ## TODO: replace with the step_index of the snapshot after NUM_PERIODS wave periods, once known
    final_step_index = 0
    time_final, field_final = load_snapshot(step_index=final_step_index)
    coord_1_final, coord_2_final = hodogram_coords(field=field_final, background=background, basis=basis)

    fig, ax = manage_plots.create_figure(fig_scale=1.4)
    ax.plot(coord_1_initial, coord_2_initial, color="grey", linestyle="--", linewidth=1.2, label=rf"$t={time_initial:.0f}$")
    ax.scatter(coord_1_final, coord_2_final, s=10, color="#c7522a", label=rf"$t={time_final:.2f}$")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_\perp \cdot \hat{e}_1$")
    ax.set_ylabel(r"$b_\perp \cdot \hat{e}_2$")

    radius_initial = numpy.hypot(coord_1_initial, coord_2_initial)
    radius_final = numpy.hypot(coord_1_final, coord_2_final)
    fractional_spread = (radius_final.max() - radius_final.min()) / radius_final.mean()
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.05,
        y_pos=0.05,
        x_alignment="left",
        y_alignment="bottom",
        label=rf"$(r_\mathrm{{max}} - r_\mathrm{{min}}) / \bar{{r}} = {fractional_spread:.2e}$",
        text_size=13,
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
