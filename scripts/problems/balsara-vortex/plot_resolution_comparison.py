## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import json
from pathlib import Path

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.mask_2d_arrays import QuadrantMasks2D
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/balsara-vortex"
DATASET_SLICE_GLOB = "magnetic_energy-slice=x_2-index=*.npz"
DATASET_TIME_NAME = "magnetic_energy-vi_evolution.json"
FIGURE_PATH = ROOT_DIR / "figures/problems/balsara-vortex/resolution-comparison.png"

RESOLUTIONS = (64, 128)

PALETTE_NAME = "cmr.horizon_r"
PALETTE_RANGE = (0.0, 1.0)
VALUE_RANGE = (-10.3, -4.3)

AXIS_BOUNDS: plot_data.AxisBounds = ((-5.0, 5.0), (-5.0, 5.0))
REFERENCE_RADIUS = 2.5
NUM_ORBITS = 3

##
## === HELPER FUNCTIONS
##


def find_slice_paths(
    *,
    data_dir: Path,
) -> list[Path]:
    """Return every slice in `data_dir`, sorted earliest (t = 0) to latest."""
    slice_paths = sorted(
        data_dir.glob(DATASET_SLICE_GLOB),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{DATASET_SLICE_GLOB}` found in: {data_dir}")
    return slice_paths


def upsample_slice(
    *,
    array_2d: NDArray[numpy.floating],
    target_num_cells: int,
) -> NDArray[numpy.floating]:
    """Block-replicate a coarser array up to `target_num_cells` (not interpolation)."""
    num_rows, num_cols = array_2d.shape
    if (num_rows == target_num_cells) and (num_cols == target_num_cells):
        return array_2d
    scale_row = target_num_cells // num_rows
    scale_col = target_num_cells // num_cols
    return numpy.kron(array_2d, numpy.ones((scale_row, scale_col)))


def compute_centroid_position(
    *,
    array_2d: NDArray[numpy.floating],
    axis_bounds: plot_data.AxisBounds,
) -> tuple[float, float]:
    """Mass-weighted centroid of a positive-definite field, in physical (x, y) units."""
    num_rows, num_cols = array_2d.shape
    cell_size_x = (axis_bounds[0][1] - axis_bounds[0][0]) / num_cols
    cell_size_y = (axis_bounds[1][1] - axis_bounds[1][0]) / num_rows
    x_coords = axis_bounds[0][0] + cell_size_x * (numpy.arange(num_cols) + 0.5)
    y_coords = axis_bounds[1][0] + cell_size_y * (numpy.arange(num_rows) + 0.5)
    x_grid, y_grid = numpy.meshgrid(x_coords, y_coords)
    total = array_2d.sum()
    return (
        float((x_grid * array_2d).sum() / total),
        float((y_grid * array_2d).sum() / total),
    )


def recenter_via_periodic_shift(
    *,
    array_2d: NDArray[numpy.floating],
    axis_bounds: plot_data.AxisBounds,
) -> NDArray[numpy.floating]:
    """
    Undo a small, uniform positional drift by applying a sub-pixel periodic shift.

    While the advecting vortex travels an exact integer number of domain-lengths, and therefore
    returns to its starting position, there is a small residual offset due to numerical dispersion effects,
    which we remove via a Fourier-space shift, which is exact for periodic, band-limited data.
    """
    centroid_x, centroid_y = compute_centroid_position(
        array_2d=array_2d,
        axis_bounds=axis_bounds,
    )
    num_rows, num_cols = array_2d.shape
    cell_size_x = (axis_bounds[0][1] - axis_bounds[0][0]) / num_cols
    cell_size_y = (axis_bounds[1][1] - axis_bounds[1][0]) / num_rows
    shift_in_cells_x = -centroid_x / cell_size_x
    shift_in_cells_y = -centroid_y / cell_size_y
    freq_x = numpy.fft.fftfreq(num_cols)
    freq_y = numpy.fft.fftfreq(num_rows)
    phase_ramp = numpy.exp(
        -2.0j * numpy.pi * (freq_x[None, :] * shift_in_cells_x + freq_y[:, None] * shift_in_cells_y),
    )
    return numpy.fft.ifft2(numpy.fft.fft2(array_2d) * phase_ramp).real


def estimate_energy_conservation_per_orbit(
    *,
    data_dir: Path,
) -> float:
    """
    Per-orbit fraction of the initial (volume-integrated) magnetic energy retained.

    Energy loss compounds geometrically orbit-to-orbit, so the per-orbit rate is the `NUM_ORBITS`-th
    root of the total retained fraction, not that total fraction divided by `NUM_ORBITS`.
    """
    vi_path = data_dir / DATASET_TIME_NAME
    with vi_path.open() as file:
        vi_data = json.load(file)
    vi_values = vi_data["vi_values"]
    total_retention_fraction = vi_values[-1] / vi_values[0]
    return total_retention_fraction**(1.0 / NUM_ORBITS)


def plot_slice_quadrants(
    *,
    top_left: NDArray[numpy.floating],
    top_right: NDArray[numpy.floating],
    bottom_left: NDArray[numpy.floating],
    bottom_right: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    num_rows, num_cols = top_left.shape
    Corner = box_positions.Positions.Corner
    composite = numpy.zeros_like(top_left)
    for array_2d, anchor in (
        (top_left, Corner.TopLeft),
        (top_right, Corner.TopRight),
        (bottom_left, Corner.BottomLeft),
        (bottom_right, Corner.BottomRight),
    ):
        mask = QuadrantMasks2D.get_mask(
            num_rows=num_rows,
            num_cols=num_cols,
            anchor=anchor,
        )
        composite = numpy.where(mask, array_2d, composite)
    return composite


def add_reference_circle(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    theta = numpy.linspace(0.0, 2.0 * numpy.pi, 200)
    ax.plot(
        REFERENCE_RADIUS * numpy.cos(theta),
        REFERENCE_RADIUS * numpy.sin(theta),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )


def add_advection_arrow(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    direction_component = 1.0 / numpy.sqrt(2.0)
    arrow_start_radius = REFERENCE_RADIUS
    arrow_end_radius = REFERENCE_RADIUS + 1.25
    label_anchor_radius = REFERENCE_RADIUS + 0.2
    label_anchor = label_anchor_radius * direction_component
    label_offset = 0.35
    advection_label_offset = label_offset + 0.15
    ax.annotate(
        "",
        xy=(arrow_end_radius * direction_component, arrow_end_radius * direction_component),
        xytext=(arrow_start_radius * direction_component, arrow_start_radius * direction_component),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linestyle": "-",
            "linewidth": 1.0,
            "mutation_scale": 15.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    ax.text(
        label_anchor - advection_label_offset,
        label_anchor + advection_label_offset,
        "advection\ndirection",
        ha="left",
        va="center",
        multialignment="left",
        rotation=45.0,
        rotation_mode="anchor",
        fontsize=24,
    )
    ax.text(
        label_anchor + label_offset,
        label_anchor - label_offset,
        r"$\mathcal{M} = 0.01$",
        ha="left",
        va="center",
        rotation=45.0,
        rotation_mode="anchor",
        fontsize=24,
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
    data_dirs_lookup = {
        num_cells: DATASET_DIR / f"ncells={num_cells}/q26-b25-ppm_ep/extracted"
        for num_cells in RESOLUTIONS
    }
    highest_resolution = max(RESOLUTIONS)
    left_side_resolution, right_side_resolution = RESOLUTIONS
    first_snapshot_lookup = {
        num_cells:
        upsample_slice(
            array_2d=numpy.load(find_slice_paths(data_dir=data_dir)[0])["sarray_2d"],
            target_num_cells=highest_resolution,
        )
        for num_cells, data_dir in data_dirs_lookup.items()
    }
    final_snapshot_lookup = {
        num_cells:
        upsample_slice(
            array_2d=recenter_via_periodic_shift(
                array_2d=numpy.load(find_slice_paths(data_dir=data_dir)[-1])["sarray_2d"],
                axis_bounds=AXIS_BOUNDS,
            ),
            target_num_cells=highest_resolution,
        )
        for num_cells, data_dir in data_dirs_lookup.items()
    }
    energy_conservation_lookup = {
        num_cells: estimate_energy_conservation_per_orbit(data_dir=data_dir)
        for num_cells, data_dir in data_dirs_lookup.items()
    }
    composite = plot_slice_quadrants(
        top_left=final_snapshot_lookup[left_side_resolution],
        top_right=final_snapshot_lookup[right_side_resolution],
        bottom_left=first_snapshot_lookup[left_side_resolution],
        bottom_right=first_snapshot_lookup[right_side_resolution],
    )
    fig, ax = manage_plots.create_figure(
        axis_shape=(7, 7),
    )
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=compute_array_stats.compute_safe_log10(composite),
        data_format="ij",
        axis_bounds=AXIS_BOUNDS,
        cbar_bounds=VALUE_RANGE,
        palette_config=add_color.SequentialConfig(
            palette_name=PALETTE_NAME,
            palette_range=PALETTE_RANGE,
        ),
        add_cbar=False,
    )
    ax.axhline(
        0.0,
        color="black",
        linewidth=0.6,
    )
    ax.axvline(
        0.0,
        color="black",
        linewidth=0.6,
    )
    add_reference_circle(ax=ax)
    add_advection_arrow(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for x_pos, num_cells in (
        (0.025, left_side_resolution),
        (0.975, right_side_resolution),
    ):
        annotate_axis.add_text(
            ax=ax,
            x_pos=x_pos,
            y_pos=0.965,
            label=rf"${num_cells}^2$",
            x_alignment=box_positions.Positions.Side.Left
            if x_pos < 0.5 else box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=24,
            text_color="black",
            box_alpha=0.0,
        )
    for x_pos, num_cells in ((0.025, left_side_resolution), (0.975, right_side_resolution)):
        annotate_axis.add_text(
            ax=ax,
            x_pos=x_pos,
            y_pos=0.025,
            label=f"conserves\n{100.0 * energy_conservation_lookup[num_cells]:.1f}\\% / orbit",
            x_alignment=(
                box_positions.Positions.Side.Left if x_pos < 0.5 else box_positions.Positions.Side.Right
            ),
            y_alignment=box_positions.Positions.Side.Bottom,
            text_size=24,
            text_color="black",
            box_alpha=0.0,
        )
    ax.text(
        0.5,
        1.02,
        "initial profile",
        transform=ax.transAxes,
        fontsize=32,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        -0.02,
        "profile after 3 orbits",
        transform=ax.transAxes,
        fontsize=32,
        ha="center",
        va="top",
    )
    palette = add_color.make_palette(
        config=add_color.SequentialConfig(
            palette_name=PALETTE_NAME,
            palette_range=PALETTE_RANGE,
        ),
        value_range=VALUE_RANGE,
    )
    add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$\log_{10}(b^2 / 2)$",
        cbar_side="right",
        label_size=32,
        label_pad=24.0,
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
