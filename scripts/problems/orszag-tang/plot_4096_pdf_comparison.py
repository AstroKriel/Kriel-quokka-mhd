## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import glob
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import manage_plots, style_plots

##
## === CONFIGURATION
##

## fixed compute + averaging (Q26 + Balsara2025b, our recommended scheme); only interpolation
## varies between the two 4096^2 runs available, so this isolates PPM vs PPM-EP directly
## PPM-EP plotted first (behind), PPM plotted last (in front)
INTERPOLATIONS = ("ppm_ep", "ppm")
## PPM coloured to match the Q26 EMF-reconstruction scheme in the Brio-Wu/RJ2a shock-tube figures
INTERPOLATION_COLORS = {"ppm": "gold", "ppm_ep": "black"}

## crop the low-signal tail below this value, where only a handful of near-zero-current cells
## from the two quiescent lobes contribute and the PDF estimate is dominated by binning noise
X_LIMITS = (-10.0, 0.0)

FIELD_NAME = "current_density_magnitude"
NUM_CELLS = 4096
## the computational domain is 1 x 1 in dimensionless units, matching the other OT figures, which
## plot Delta x |curl b| rather than the raw (dimensional) current density
DOMAIN_LENGTH = 1.0
CELL_SIZE = DOMAIN_LENGTH / NUM_CELLS
LOG10_CELL_SIZE = numpy.log10(CELL_SIZE)

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / f"datasets/problems/orszag-tang/ncells={NUM_CELLS}"
FIGURE_PATH = ROOT_DIR / f"figures/problems/orszag-tang/ncells={NUM_CELLS}/pdf_comparison.png"

##
## === HELPER FUNCTIONS
##


def load_final_pdf(
    *,
    interpolation: str,
) -> dict:
    """Load the last-saved-time PDF file for `q26-b25-<interpolation>` (t = 1.0)."""
    diagnostics_dir = DATASET_DIR / f"q26-b25-{interpolation}" / "diagnostics"
    pdf_paths = glob.glob(str(diagnostics_dir / f"{FIELD_NAME}-pdf-index=*.json"))
    if not pdf_paths:
        raise FileNotFoundError(f"no PDF files matching `{FIELD_NAME}-pdf-index=*.json` found in: {diagnostics_dir}")
    pdf_dicts = [json_io.read_json_file_into_dict(Path(path), verbose=False) for path in pdf_paths]
    return max(pdf_dicts, key=lambda pdf_dict: pdf_dict["step_time"])


def get_field_comp_key(
    *,
    pdf_dict: dict,
) -> str:
    """Return the one key in a PDF dict that is not run metadata (the field's math-mode label)."""
    meta_keys = {"step_time", "step_index", "log10_binning"}
    comp_keys = [key for key in pdf_dict if key not in meta_keys]
    if len(comp_keys) != 1:
        raise ValueError(f"expected exactly one field component in a scalar-field PDF file, got: {comp_keys}")
    return comp_keys[0]


def get_pdf_curve(
    *,
    pdf_dict: dict,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return (bin_centers, log10_density), shifted into Delta x |curl b| units and with empty bins dropped."""
    comp_key = get_field_comp_key(pdf_dict=pdf_dict)
    if not pdf_dict["log10_binning"]:
        raise ValueError("this script assumes PDFs were computed with log10-space binning.")
    raw_bin_centers = pdf_dict[comp_key]["bin_centers"]
    raw_log10_density = pdf_dict[comp_key]["log10_density"]
    ## masked (empty) bins are serialised as `null`/`None`; drop them rather than treating them as zero
    is_populated = [value is not None for value in raw_log10_density]
    bin_centers = numpy.asarray([value for value, keep in zip(raw_bin_centers, is_populated) if keep])
    log10_density = numpy.asarray([value for value, keep in zip(raw_log10_density, is_populated) if keep])
    return bin_centers + LOG10_CELL_SIZE, log10_density


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
        axis_shape=(4, 6),
    )
    ax = axs[0, 0]
    for interpolation in INTERPOLATIONS:
        pdf_dict = load_final_pdf(interpolation=interpolation)
        bin_centers, log10_density = get_pdf_curve(pdf_dict=pdf_dict)
        ax.step(
            bin_centers,
            log10_density,
            where="mid",
            color=INTERPOLATION_COLORS[interpolation],
            linewidth=1.8,
        )
    ax.set_xlim(X_LIMITS)
    ax.set_xlabel(r"$x \equiv \log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$")
    ax.set_ylabel(r"$\log_{10} \big( \mathrm{PDF}(x) \big)$")
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=400,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
