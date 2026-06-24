## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import tomllib

from pathlib import Path

## third-party
import numpy

from matplotlib.cm import ScalarMappable as mpl_ScalarMappable

## personal
from jormi.ww_plots import add_color, manage_plots, plot_data

##
## === CONFIGURATION
##

## scheme tokens, in the order they appear in each dataset directory name:
##     <emf-reconstruction>-<emf-averaging>-<interpolation>
EMF_RECONSTRUCTIONS = ("fs17", "b25", "q26")  ## grid columns (left -> right)
INTERPOLATIONS = ("plm", "ppm", "ppm_ep")  ## grid row-blocks (top -> bottom)
EMF_AVERAGINGS = ("ld04", "b25")  ## the two rows within each interpolation block

## the out-of-plane current density slice, shared across the whole grid
SLICE_GLOB = "cur-slice=x_2-index=*.npy"
FIELD_LABEL = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
PALETTE_NAME = "cmr.wildfire"

## floor for the shared colour scale, in log10 units of the cell-size-normalised field; the bottom
## palette colour maps to this value instead of the data minimum, so the sparsely-populated low tail
## saturates rather than stretching the scale (the field rarely drops this low)
VALUE_FLOOR = -5.0

## the computational domain is 1 x 1 in dimensionless units (see sim_params.toml)
AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/orszag-tang/ncells=1024"
FIGURE_PATH = ROOT_DIR / "figures/problems/orszag-tang/ot-schemes.pdf"

##
## === HELPER FUNCTIONS
##


def find_last_slice_path(
    *,
    scheme_dir: Path,
) -> Path:
    """Return the latest slice (the largest saved time index, t = 1.0) in `scheme_dir`."""
    slice_paths = sorted(
        scheme_dir.glob(SLICE_GLOB),
        key=lambda path: int(path.stem.split("index=")[-1]),
    )
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {scheme_dir}")
    return slice_paths[-1]


def read_cell_size(
    *,
    dataset_dir: Path,
) -> float:
    """Read the isotropic cell size dx = L / N from a representative sim_params.toml.

    Every scheme directory shares the same geometry and resolution, so any one suffices.
    """
    params_paths = sorted(dataset_dir.glob("*/sim_params.toml"))
    if not params_paths:
        raise FileNotFoundError(f"no sim_params.toml found under: {dataset_dir}")
    params = tomllib.loads(params_paths[0].read_text())
    domain_length = params["geometry"]["prob_hi"][0] - params["geometry"]["prob_lo"][0]
    num_cells = params["amr"]["n_cell"][0]
    return domain_length / num_cells


def load_scheme_slices() -> dict[tuple[str, str, str], numpy.ndarray]:
    """Load the last current-density slice for every (reconstruction, averaging, interpolation) combo."""
    slices: dict[tuple[str, str, str], numpy.ndarray] = {}
    for reconstruction in EMF_RECONSTRUCTIONS:
        for interpolation in INTERPOLATIONS:
            for averaging in EMF_AVERAGINGS:
                scheme_dir = DATASET_DIR / f"{reconstruction}-{averaging}-{interpolation}"
                slice_path = find_last_slice_path(scheme_dir=scheme_dir)
                slices[(reconstruction, averaging, interpolation)] = numpy.load(slice_path)
    return slices


def compute_shared_value_range(
    *,
    slices: dict[tuple[str, str, str], numpy.ndarray],
) -> tuple[float, float]:
    """Compute one (min, max) range across all panels so the grid shares a single colour scale.

    Only finite values are considered, so non-finite floor pixels (e.g. log10 of a zero-current
    cell, which is -inf) do not collapse the range.
    """
    finite_values = numpy.concatenate([
        array_2d[numpy.isfinite(array_2d)].ravel() for array_2d in slices.values()
    ])
    return (
        float(finite_values.min()),
        float(finite_values.max()),
    )


def add_grid_colorbar(
    *,
    fig,
    axs,
    value_range: tuple[float, float],
    gap: float = 0.02,
    thickness: float = 0.04,
) -> None:
    """Add a single colorbar spanning the full height of the grid's right-hand column."""
    palette = add_color.make_palette(
        config=add_color.SequentialConfig(palette_name=PALETTE_NAME),
        value_range=value_range,
    )
    top_box = axs[0, -1].get_position()
    bottom_box = axs[-1, -1].get_position()
    cbar_ax = fig.add_axes((
        top_box.x1 + gap,
        bottom_box.y0,
        thickness,
        top_box.y1 - bottom_box.y0,
    ))
    mappable = mpl_ScalarMappable(
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
    )
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable=mappable,
        cax=cbar_ax,
    )
    cbar.set_label(
        label=FIELD_LABEL,
        rotation=90,
    )

##
## === PROGRAM MAIN
##


def main() -> None:
    cell_size = read_cell_size(dataset_dir=DATASET_DIR)
    slices = load_scheme_slices()
    slices = {scheme: numpy.log10(cell_size * array_2d) for scheme, array_2d in slices.items()}
    _, max_value = compute_shared_value_range(slices=slices)
    value_range = (-4, -1)
    row_schemes = [
        (interpolation, averaging)
        for interpolation in INTERPOLATIONS
        for averaging in EMF_AVERAGINGS
    ]
    fig, axs = manage_plots.create_figure_grid(
        num_rows=len(row_schemes),
        num_cols=len(EMF_RECONSTRUCTIONS),
        axis_shape=(4, 4),
        x_spacing=0.02,
        y_spacing=0.02,
        share_x=True,
        share_y=True,
    )
    for row_index, (interpolation, averaging) in enumerate(row_schemes):
        for col_index, reconstruction in enumerate(EMF_RECONSTRUCTIONS):
            ax = axs[row_index, col_index]
            plot_data.plot_2d_array(
                ax=ax,
                array_2d=slices[(reconstruction, averaging, interpolation)],
                data_format="xy",
                axis_bounds=AXIS_BOUNDS,
                cbar_bounds=value_range,
                palette_config=add_color.SequentialConfig(palette_name=PALETTE_NAME),
                add_cbar=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    add_grid_colorbar(
        fig=fig,
        axs=axs,
        value_range=value_range,
    )
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
