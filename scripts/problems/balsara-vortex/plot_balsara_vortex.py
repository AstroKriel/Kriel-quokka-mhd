## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import json
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.mask_2d_arrays import QuadrantMasks2D
from jormi.ww_io import manage_io
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

RESOLUTIONS = (64, 128)  ## 64^2 -> left half, 128^2 -> right half
RESOLUTION_LABELS = {64: r"$64^2$", 128: r"$128^2$"}
## common canvas resolution the two datasets are combined onto (the coarser dataset is upsampled
## by block replication, since it is only ever shown clipped to its own quadrant, never resampled)
COMMON_NUM_CELLS = max(RESOLUTIONS)

SLICE_GLOB = "magnetic_energy-slice=x_2-index=*.npz"
FIELD_LABEL = r"$\log_{10}(b^2 / 2)$"
PALETTE_NAME = "cmr.horizon_r"
PALETTE_RANGE = (0.0, 1.0)
VALUE_RANGE = (-10.3, -4.3)

## the vortex is centred on the domain and has a characteristic core radius of 1 (\citet{Balsara04a})
AXIS_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))
VORTEX_CORE_RADIUS = 2.5

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/balsara-vortex"
FIGURE_PATH = ROOT_DIR / "figures/problems/balsara-vortex/balsara-vortex.png"

##
## === HELPER FUNCTIONS
##


def find_first_slice_path(
    *,
    diagnostics_dir: Path,
) -> Path:
    """Return the earliest slice (t = 0, the initial condition) in `diagnostics_dir`."""
    slice_paths = sorted(
        diagnostics_dir.glob(SLICE_GLOB),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {diagnostics_dir}")
    return slice_paths[0]


def find_last_slice_path(
    *,
    diagnostics_dir: Path,
) -> Path:
    """Return the latest slice (t = 30 sqrt(2), after 3 diagonal crossings) in `diagnostics_dir`."""
    slice_paths = sorted(
        diagnostics_dir.glob(SLICE_GLOB),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{SLICE_GLOB}` found in: {diagnostics_dir}")
    return slice_paths[-1]


def upsample_by_block_replication(
    *,
    array_2d: numpy.ndarray,
    target_num_cells: int,
) -> numpy.ndarray:
    """Block-replicate a coarser array up to `target_num_cells`, with no interpolation."""
    num_rows, num_cols = array_2d.shape
    if (num_rows == target_num_cells) and (num_cols == target_num_cells):
        return array_2d
    scale_row = target_num_cells // num_rows
    scale_col = target_num_cells // num_cols
    return numpy.kron(array_2d, numpy.ones((scale_row, scale_col)))


def compute_centroid(
    *,
    array_2d: numpy.ndarray,
    axis_bounds: tuple[tuple[float, float], tuple[float, float]],
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
    array_2d: numpy.ndarray,
    axis_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> numpy.ndarray:
    """Undo a small, uniform positional drift by applying a sub-pixel periodic shift.

    The vortex is advected an exact integer number of domain-lengths, so it should return to its
    initial centroid; any residual offset (a small numerical dispersion effect, not a resolution
    dependent one) is removed here via a Fourier-space shift, exact for periodic, band-limited data.
    """
    centroid_x, centroid_y = compute_centroid(array_2d=array_2d, axis_bounds=axis_bounds)
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


## the vortex advects diagonally across the periodic domain this many times over the run
NUM_ORBITS = 3


def compute_retention_fraction_per_orbit(
    *,
    diagnostics_dir: Path,
) -> float:
    """Per-orbit fraction of the initial (volume-integrated) magnetic energy retained.

    Energy loss compounds geometrically orbit-to-orbit, so the per-orbit rate is the `NUM_ORBITS`-th
    root of the total retained fraction, not that total fraction divided by `NUM_ORBITS`.
    """
    vi_path = diagnostics_dir / "magnetic_energy-vi_evolution.json"
    with vi_path.open() as file:
        vi_data = json.load(file)
    vi_values = vi_data["vi_values"]
    total_retention_fraction = vi_values[-1] / vi_values[0]
    return total_retention_fraction ** (1.0 / NUM_ORBITS)


def compose_quadrants(
    *,
    top_left: numpy.ndarray,
    top_right: numpy.ndarray,
    bottom_left: numpy.ndarray,
    bottom_right: numpy.ndarray,
) -> numpy.ndarray:
    """Combine four arrays into one, each clipped to its own corner via `QuadrantMasks2D`."""
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
    ax,
) -> None:
    theta = numpy.linspace(0.0, 2.0 * numpy.pi, 200)
    ax.plot(
        VORTEX_CORE_RADIUS * numpy.cos(theta),
        VORTEX_CORE_RADIUS * numpy.sin(theta),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )


def add_advection_arrow(
    *,
    ax,
) -> None:
    direction_component = 1.0 / numpy.sqrt(2.0)
    arrow_start_radius = VORTEX_CORE_RADIUS
    arrow_end_radius = VORTEX_CORE_RADIUS + 1.25
    label_anchor_radius = VORTEX_CORE_RADIUS + 0.2
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
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    diagnostics_dirs = {
        num_cells: DATASET_DIR / f"ncells={num_cells}/q26-b25-ppm_ep/diagnostics" for num_cells in RESOLUTIONS
    }
    initial_slices = {
        num_cells: upsample_by_block_replication(
            array_2d=numpy.load(find_first_slice_path(diagnostics_dir=diagnostics_dir))["sarray_2d"],
            target_num_cells=COMMON_NUM_CELLS,
        )
        for num_cells, diagnostics_dir in diagnostics_dirs.items()
    }
    final_slices = {
        num_cells: upsample_by_block_replication(
            array_2d=recenter_via_periodic_shift(
                array_2d=numpy.load(find_last_slice_path(diagnostics_dir=diagnostics_dir))["sarray_2d"],
                axis_bounds=AXIS_BOUNDS,
            ),
            target_num_cells=COMMON_NUM_CELLS,
        )
        for num_cells, diagnostics_dir in diagnostics_dirs.items()
    }
    retention_fractions_per_orbit = {
        num_cells: compute_retention_fraction_per_orbit(diagnostics_dir=diagnostics_dir)
        for num_cells, diagnostics_dir in diagnostics_dirs.items()
    }
    left_res, right_res = RESOLUTIONS
    composite = compose_quadrants(
        top_left=final_slices[left_res],
        top_right=final_slices[right_res],
        bottom_left=initial_slices[left_res],
        bottom_right=initial_slices[right_res],
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(8, 8),
    )
    ax = axs[0, 0]
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=compute_array_stats.compute_safe_log10(composite),
        data_format="ij",
        axis_bounds=AXIS_BOUNDS,
        cbar_bounds=VALUE_RANGE,
        palette_config=add_color.SequentialConfig(palette_name=PALETTE_NAME, palette_range=PALETTE_RANGE),
        add_cbar=False,
    )
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    add_reference_circle(ax=ax)
    add_advection_arrow(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for x_pos, num_cells in ((0.025, left_res), (0.975, right_res)):
        annotate_axis.add_text(
            ax=ax,
            x_pos=x_pos,
            y_pos=0.965,
            label=RESOLUTION_LABELS[num_cells],
            x_alignment=box_positions.Positions.Side.Left if x_pos < 0.5 else box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=24,
            text_color="black",
            box_alpha=0.0,
        )
    for x_pos, num_cells in ((0.025, left_res), (0.975, right_res)):
        annotate_axis.add_text(
            ax=ax,
            x_pos=x_pos,
            y_pos=0.025,
            label=f"conserve\n{100.0 * retention_fractions_per_orbit[num_cells]:.1f}\\% / orbit",
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
        config=add_color.SequentialConfig(palette_name=PALETTE_NAME, palette_range=PALETTE_RANGE),
        value_range=VALUE_RANGE,
    )
    add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=FIELD_LABEL,
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
