## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
from pathlib import Path

## third-party
import numpy
from matplotlib import gridspec as mpl_gridspec
from matplotlib import patches as mpl_patches
from matplotlib import pyplot as mpl_plot

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_log
from jormi.ww_plots import add_color, manage_plots, plot_data, style_plots

##
## === CONFIGURATION
##

ROOT_DIR = Path(__file__).parents[3]

FIELD_LABEL = r"$\log_{10} \left( \Delta x \, |\nabla \times \vec{b}| \right)$"
FIELD_PALETTE = "cmr.wildfire_r"
FIELD_PALETTE_RANGE = (0.2, 1.0)
FIELD_RANGE = (-3.0, -1.25)

AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

## ((x0, y0), (x1, y1)) per zoom panel; corners are normalised (sorted) before cropping,
## so it does not matter whether a given axis is given low-to-high or high-to-low.
ZOOM_REGIONS = (
    ((0.0525, 0.001625), (0.135, -0.080875)),
    ((0.4225, -0.0985), (0.2725, -0.2485)),
)

ZOOM_BOX_COLOR = "white"
ZOOM_BOX_LINEWIDTH = 1.0

## main panel occupies the left NUM_MAIN_CELLS x NUM_MAIN_CELLS block (kept square); the zoom
## panels stack vertically to its right, each spanning ZOOM_CELL_SIZE rows so their combined
## height matches the main panel and every panel (main and zoom) remains square
NUM_ZOOM_AXES = len(ZOOM_REGIONS)
NUM_MAIN_CELLS = 6
ZOOM_CELL_SIZE = NUM_MAIN_CELLS // NUM_ZOOM_AXES
NUM_GRID_ROWS = NUM_MAIN_CELLS
NUM_GRID_COLS = NUM_MAIN_CELLS + ZOOM_CELL_SIZE

CELL_SIZE_INCHES = 1.5

##
## === HELPER FUNCTIONS
##


def read_cell_size(
    *,
    data_path: Path,
) -> float:
    """Compute the isotropic cell size dx = L / N from the slice path's `ncells=<N>` ancestor directory.

    The computational domain is 1 x 1 in dimensionless units (see `AXIS_BOUNDS`); `sim_params.toml`
    is not read here, so `ncells` is parsed from the path (see `plot_slice.read_cell_size`).
    """
    ncells_dir = next(
        (parent for parent in data_path.parents if parent.name.startswith("ncells=")),
        None,
    )
    if ncells_dir is None:
        raise ValueError(f"no `ncells=<N>` ancestor directory found for: {data_path}")
    num_cells = int(ncells_dir.name.split("=")[-1])
    domain_length = AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]
    return domain_length / num_cells


def crop_field_to_region(
    *,
    log_field: numpy.ndarray,
    region: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[numpy.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    """Crop a full-domain `log_field` (indexed [x, y], see `data_format="xy"`) down to `region`.

    `region` corners are normalised (sorted) first, so axes may be given in either order.
    Returns the cropped array and the corresponding (normalised) axis bounds.
    """
    (x0, y0), (x1, y1) = region
    x_lo, x_hi = sorted((x0, x1))
    y_lo, y_hi = sorted((y0, y1))
    num_cells_x, num_cells_y = log_field.shape
    domain_x0, domain_x1 = AXIS_BOUNDS[0]
    domain_y0, domain_y1 = AXIS_BOUNDS[1]
    ix_lo = round((x_lo - domain_x0) / (domain_x1 - domain_x0) * num_cells_x)
    ix_hi = round((x_hi - domain_x0) / (domain_x1 - domain_x0) * num_cells_x)
    iy_lo = round((y_lo - domain_y0) / (domain_y1 - domain_y0) * num_cells_y)
    iy_hi = round((y_hi - domain_y0) / (domain_y1 - domain_y0) * num_cells_y)
    cropped_field = log_field[ix_lo:ix_hi, iy_lo:iy_hi]
    return cropped_field, ((x_lo, x_hi), (y_lo, y_hi))


def draw_zoom_box(
    *,
    ax: mpl_plot.Axes,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Outline `bounds` (already-normalised (x_lo, x_hi), (y_lo, y_hi)) on `ax` with an open rectangle."""
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    ax.add_patch(
        mpl_patches.Rectangle(
            (x_lo, y_lo),
            x_hi - x_lo,
            y_hi - y_lo,
            fill=False,
            edgecolor=ZOOM_BOX_COLOR,
            linewidth=ZOOM_BOX_LINEWIDTH,
        ),
    )


def plot_field_on_axis(
    *,
    ax: mpl_plot.Axes,
    log_field: numpy.ndarray,
    axis_bounds: tuple[tuple[float, float], tuple[float, float]],
    add_cbar: bool,
) -> None:
    field_config = add_color.SequentialConfig(palette_name=FIELD_PALETTE, palette_range=FIELD_PALETTE_RANGE)
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=log_field,
        data_format="xy",
        axis_bounds=axis_bounds,
        cbar_bounds=FIELD_RANGE,
        palette_config=field_config,
        add_cbar=add_cbar,
        cbar_label=FIELD_LABEL if add_cbar else None,
        cbar_side="top",
    )
    ax.set_xticks([])
    ax.set_yticks([])


##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an OT current-density slice with zoom-in panels.")
    parser.add_argument("data_path", type=Path, help="Path to the .npz slice file.")
    args = parser.parse_args()

    data_path = args.data_path.expanduser().resolve()
    stem = data_path.stem

    dataset_relative_dir = data_path.relative_to(ROOT_DIR / "datasets").parent
    if dataset_relative_dir.name == "extracted":
        ## raw slices are nested under a `extracted/` subdir; figures stay flat,
        ## mirroring the scheme dir itself rather than that subdir.
        dataset_relative_dir = dataset_relative_dir.parent
    figures_dir = ROOT_DIR / "figures" / dataset_relative_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / (stem + "-zoom.png")

    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    cell_size = read_cell_size(data_path=data_path)
    field = numpy.load(data_path)["sarray_2d"]
    log_field = compute_array_stats.compute_safe_log10(cell_size * numpy.abs(field))

    style_plots.set_theme(theme=style_plots.Theme.LIGHT)
    fig = mpl_plot.figure(
        figsize=(
            CELL_SIZE_INCHES * NUM_GRID_COLS,
            CELL_SIZE_INCHES * NUM_GRID_ROWS,
        ),
    )
    grid_spec = mpl_gridspec.GridSpec(nrows=NUM_GRID_ROWS, ncols=NUM_GRID_COLS, figure=fig)

    ax_main = fig.add_subplot(grid_spec[:, 0:NUM_MAIN_CELLS])
    plot_field_on_axis(ax=ax_main, log_field=log_field, axis_bounds=AXIS_BOUNDS, add_cbar=True)

    for zoom_index, region in enumerate(ZOOM_REGIONS):
        row_start = zoom_index * ZOOM_CELL_SIZE
        row_end = row_start + ZOOM_CELL_SIZE
        ax_zoom = fig.add_subplot(grid_spec[row_start:row_end, NUM_MAIN_CELLS:NUM_GRID_COLS])
        cropped_field, cropped_bounds = crop_field_to_region(log_field=log_field, region=region)
        plot_field_on_axis(ax=ax_zoom, log_field=cropped_field, axis_bounds=cropped_bounds, add_cbar=False)
        draw_zoom_box(ax=ax_main, bounds=cropped_bounds)

    manage_plots.save_figure(fig=fig, fig_path=fig_path, dpi=400)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
