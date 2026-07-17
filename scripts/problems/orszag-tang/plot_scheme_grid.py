## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

from matplotlib.cm import ScalarMappable as mpl_ScalarMappable

## personal
from jormi.ww_arrays.mask_2d_arrays import DiagonalMasks2D
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

## scheme tokens, in the order they appear in each dataset directory name:
##     <emf-reconstruction>-<emf-averaging>-<interpolation>
EMF_RECONSTRUCTIONS = ("fs17", "b25", "q26")  ## grid columns (left -> right)
INTERPOLATIONS = ("plm", "ppm", "ppm_ep")  ## grid rows (top -> bottom)
## the two EMF averaging schemes share a single panel per (reconstruction, interpolation) combo,
## split across the main diagonal: the OT field is exactly rot180-symmetric, so either half of
## the domain already contains the same information as the other, and we use the "spare" half to
## show the other averaging scheme instead of repeating the same (mirrored) data twice.
EMF_AVERAGINGS = ("ld04", "b25")

## display shorthand, distinct from the lowercase dataset-directory tokens
RECONSTRUCTION_LABELS = {"fs17": "FS17", "b25": "B25", "q26": "Q26"}
INTERPOLATION_LABELS = {"plm": "PLM", "ppm": "PPM", "ppm_ep": "PPM-EP"}
AVERAGING_LABELS = {"ld04": "LD04", "b25": "B25"}

## the out-of-plane current density slice, shared across the whole grid
SLICE_GLOB = "current_density_magnitude-slice=x_2-index=*.npz"
## compare at the second of the three saved snapshots (t = 0.5, 0.85, 1.0), rather than the last
TARGET_TIME = 0.85
FIELD_LABEL = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
## same colormap as the neighbouring `plot_slice.py`, subset to its upper half
PALETTE_NAME = "cmr.eclipse"
PALETTE_RANGE = (0.0, 1.0)

## the computational domain is 1 x 1 in dimensionless units
AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/orszag-tang/ncells=1024"
FIGURE_PATH = ROOT_DIR / "figures/problems/orszag-tang/ncells=1024/scheme_grid.png"

##
## === HELPER FUNCTIONS
##


def find_slice_near_time(
    *,
    scheme_dir: Path,
) -> Path:
    """Return the saved slice whose `step_time` is closest to `TARGET_TIME`.

    Different schemes take different numbers of (CFL-limited) timesteps to reach the same physical
    time, so the saved snapshot index that corresponds to `TARGET_TIME` differs slightly between
    combos; matching on the metadata directly (rather than assuming a shared index) is robust to that.
    """
    slice_paths = sorted((scheme_dir / "diagnostics").glob(SLICE_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {scheme_dir / 'diagnostics'}")
    return min(
        slice_paths,
        key=lambda path: abs(float(numpy.load(path)["step_time"]) - TARGET_TIME),
    )


def read_cell_size(
    *,
    dataset_dir: Path,
) -> float:
    """Compute the isotropic cell size dx = L / N from the `ncells=<N>` dataset directory name.

    The computational domain is 1 x 1 in dimensionless units (see `AXIS_BOUNDS`); `sim_params.toml`
    is not committed to the repo (raw configs stay on the HPC), so `ncells` is read from the path.
    """
    num_cells = int(dataset_dir.name.split("=")[-1])
    domain_length = AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]
    return domain_length / num_cells


def load_scheme_slices() -> dict[tuple[str, str, str], numpy.ndarray]:
    """Load the t = 0.85 current-density slice for every (reconstruction, averaging, interpolation) combo."""
    slices: dict[tuple[str, str, str], numpy.ndarray] = {}
    for reconstruction in EMF_RECONSTRUCTIONS:
        for interpolation in INTERPOLATIONS:
            for averaging in EMF_AVERAGINGS:
                scheme_dir = DATASET_DIR / f"{reconstruction}-{averaging}-{interpolation}"
                slice_path = find_slice_near_time(scheme_dir=scheme_dir)
                slices[(reconstruction, averaging, interpolation)] = numpy.load(slice_path)["sarray_2d"]
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


def compose_averaging_split(
    *,
    upper_array: numpy.ndarray,
    lower_array: numpy.ndarray,
) -> numpy.ndarray:
    """Combine two (rot180-symmetric) fields into one array, split across the main diagonal.

    Neither half loses information relative to showing either field in full: each field already
    repeats itself (point-reflected) between the two halves, so one half per field is sufficient.
    """
    num_rows, num_cols = upper_array.shape
    upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return numpy.where(upper_mask, upper_array, lower_array)


def add_grid_colorbar(
    *,
    fig,
    axs,
    value_range: tuple[float, float],
    gap: float = 0.015,
    thickness: float = 0.0225,
) -> None:
    """Add a single vertical colorbar spanning the full height of the grid's right-hand column."""
    palette = add_color.make_palette(
        config=add_color.SequentialConfig(palette_name=PALETTE_NAME, palette_range=PALETTE_RANGE),
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
    style_plots.set_theme()
    cell_size = read_cell_size(dataset_dir=DATASET_DIR)
    slices = load_scheme_slices()
    slices = {scheme: numpy.log10(cell_size * array_2d) for scheme, array_2d in slices.items()}
    value_range = (-2.5, -1.2)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=len(INTERPOLATIONS),
        num_cols=len(EMF_RECONSTRUCTIONS),
        axis_shape=(4, 4.35),
        x_spacing=0.02,
        y_spacing=0.02,
        share_x=True,
        share_y=True,
    )
    for row_index, interpolation in enumerate(INTERPOLATIONS):
        for col_index, reconstruction in enumerate(EMF_RECONSTRUCTIONS):
            ax = axs[row_index, col_index]
            composite = compose_averaging_split(
                upper_array=slices[(reconstruction, EMF_AVERAGINGS[0], interpolation)],
                lower_array=slices[(reconstruction, EMF_AVERAGINGS[1], interpolation)],
            )
            plot_data.plot_2d_array(
                ax=ax,
                array_2d=composite,
                data_format="xy",
                axis_bounds=AXIS_BOUNDS,
                cbar_bounds=value_range,
                palette_config=add_color.SequentialConfig(palette_name=PALETTE_NAME, palette_range=PALETTE_RANGE),
                add_cbar=False,
            )
            ax.plot(
                [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
                [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
                color="white",
                linewidth=0.6,
            )
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.05,
                y_pos=0.95,
                label=AVERAGING_LABELS[EMF_AVERAGINGS[0]],
                x_alignment=box_positions.Positions.Side.Left,
                y_alignment=box_positions.Positions.Side.Top,
                text_size=22,
                text_color="white",
                box_color="black",
                box_alpha=0.85,
            )
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.95,
                y_pos=0.05,
                label=AVERAGING_LABELS[EMF_AVERAGINGS[1]],
                x_alignment=box_positions.Positions.Side.Right,
                y_alignment=box_positions.Positions.Side.Bottom,
                text_size=22,
                text_color="white",
                box_color="black",
                box_alpha=0.85,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(RECONSTRUCTION_LABELS[reconstruction])
            if col_index == 0:
                ax.set_ylabel(INTERPOLATION_LABELS[interpolation])
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
