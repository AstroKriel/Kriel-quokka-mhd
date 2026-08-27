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
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

## dataset paths and file layout
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/balsara-vortex"
DATASET_RESOLUTIONS = (64, 128)
DATASET_SLICE_GLOB = "magnetic_energy-slice=x_2-index=*.npz"
DATASET_TIME_NAME = "magnetic_energy-vi_evolution.json"
FIGURE_PATH = ROOT_DIR / "figures/problems/balsara-vortex/resolution-comparison.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-5.0, 5.0), (-5.0, 5.0))
PALETTE_NAME = "cmr.horizon_r"
PALETTE_RANGE = (0.0, 1.0)
VALUE_RANGE = (-10, -3.5)

## annotations
REFERENCE_RADIUS = 2.0
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
    axis_ranges: plot_data.AxisRanges,
) -> tuple[float, float]:
    """Mass-weighted centroid of a positive-definite field, in physical (x, y) units."""
    num_rows, num_cols = array_2d.shape
    cell_size_x = (axis_ranges[0][1] - axis_ranges[0][0]) / num_cols
    cell_size_y = (axis_ranges[1][1] - axis_ranges[1][0]) / num_rows
    x_coords = axis_ranges[0][0] + cell_size_x * (numpy.arange(num_cols) + 0.5)
    y_coords = axis_ranges[1][0] + cell_size_y * (numpy.arange(num_rows) + 0.5)
    x_grid, y_grid = numpy.meshgrid(x_coords, y_coords)
    total = array_2d.sum()
    return (
        float((x_grid * array_2d).sum() / total),
        float((y_grid * array_2d).sum() / total),
    )


def recenter_via_periodic_shift(
    *,
    array_2d: NDArray[numpy.floating],
    axis_ranges: plot_data.AxisRanges,
) -> NDArray[numpy.floating]:
    """
    Undo a small, uniform positional drift by applying a sub-pixel periodic shift.

    While the advecting vortex travels an exact integer number of domain-lengths, and therefore
    returns to its starting position, there is a small residual offset due to numerical dispersion effects,
    which we remove via a Fourier-space shift, which is exact for periodic, band-limited data.
    """
    centroid_x, centroid_y = compute_centroid_position(
        array_2d=array_2d,
        axis_ranges=axis_ranges,
    )
    num_rows, num_cols = array_2d.shape
    cell_size_x = (axis_ranges[0][1] - axis_ranges[0][0]) / num_cols
    cell_size_y = (axis_ranges[1][1] - axis_ranges[1][0]) / num_rows
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
    data_path = data_dir / DATASET_TIME_NAME
    with data_path.open() as file:
        dataset = json.load(file)
    dataset_values = dataset["vi_values"]
    energy_conservation_fraction = dataset_values[-1] / dataset_values[0]
    return energy_conservation_fraction**(1.0 / NUM_ORBITS)


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


def add_reference_circle_and_drift_arrow(
    *,
    panel: manage_figure.Panel,
    radius: float,
) -> None:
    """Draw the reference circle and the drift-direction arrow, both anchored to `radius`."""
    figure_params = style_figure.get_figure_params()
    artist_params = figure_params.artist_params
    text_size_params = figure_params.text_size_params
    theme_params = figure_params.theme_params
    theta = numpy.linspace(0.0, 2.0 * numpy.pi, 200)
    panel.plot(
        radius * numpy.cos(theta),
        radius * numpy.sin(theta),
        color=theme_params.foreground_color,
        linestyle="--",
    )
    direction_component = 1.0 / numpy.sqrt(2.0)
    arrow_start_radius = radius
    arrow_end_radius = radius + 1.25
    label_anchor_radius = radius + 0.2
    label_anchor = label_anchor_radius * direction_component
    label_offset = 0.35
    drift_label_offset = label_offset + 0.25
    panel.annotate(
        "",
        xy=(arrow_end_radius * direction_component, arrow_end_radius * direction_component),
        xytext=(arrow_start_radius * direction_component, arrow_start_radius * direction_component),
        arrowprops={
            "arrowstyle": "-|>",
            "color": theme_params.foreground_color,
            "linestyle": "-",
            "linewidth": artist_params.line_width_pt,
            ## the head is sized in points, so tie it to the text it sits beside
            "mutation_scale": text_size_params.annotation_size,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    panel.text(
        label_anchor - drift_label_offset,
        label_anchor + drift_label_offset,
        "drift\ndirection",
        ha="left",
        va="center",
        multialignment="left",
        rotation=45.0,
        rotation_mode="anchor",
        fontsize=text_size_params.annotation_size,
        color=theme_params.foreground_color,
    )
    panel.text(
        label_anchor + label_offset,
        label_anchor - label_offset,
        r"$\mathcal{M} = 0.01$",
        ha="left",
        va="center",
        rotation=45.0,
        rotation_mode="anchor",
        fontsize=text_size_params.annotation_size,
        color=theme_params.foreground_color,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    figure_params = style_figure.get_figure_params()
    frame_params = figure_params.frame_params
    panel_gaps = figure_params.figure_layout.panel_gaps
    text_size_params = figure_params.text_size_params
    theme_params = figure_params.theme_params
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    data_dirs_lookup = {
        num_cells: DATASET_DIR / f"ncells={num_cells}/q26-b25-ppm_ep/extracted"
        for num_cells in DATASET_RESOLUTIONS
    }
    highest_resolution = max(DATASET_RESOLUTIONS)
    left_side_resolution, right_side_resolution = DATASET_RESOLUTIONS
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
                axis_ranges=AXIS_BOUNDS,
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
    figure, panel = manage_figure.create_figure(
        ## the domain is square; the figure is fitted around it once its labels exist
        panel_aspect_ratio=1.0,
        ## drawn at the width the paper prints it at, so its text is the size it asks for
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=compute_array_stats.compute_safe_log10(composite),
        data_format="ij",
        axis_ranges=AXIS_BOUNDS,
        colorbar_range=VALUE_RANGE,
        palette_config=add_color.SequentialConfig(
            palette_name=PALETTE_NAME,
            palette_range=PALETTE_RANGE,
        ),
        add_colorbar=False,
    )
    panel.axhline(
        0.0,
        color=theme_params.foreground_color,
        linewidth=frame_params.line_width_pt,
    )
    panel.axvline(
        0.0,
        color=theme_params.foreground_color,
        linewidth=frame_params.line_width_pt,
    )
    add_reference_circle_and_drift_arrow(
        panel=panel,
        radius=REFERENCE_RADIUS,
    )
    panel.set_xticks([])
    panel.set_yticks([])
    for x_pos, num_cells in (
        (0.025, left_side_resolution),
        (0.975, right_side_resolution),
    ):
        annotate_panel.add_text(
            panel=panel,
            x_pos_fraction=x_pos,
            y_pos_fraction=0.965,
            label=rf"${num_cells}^2$",
            x_alignment=box_positions.Positions.Side.Left
            if x_pos < 0.5 else box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Top,
            ## these name what each half of the panel shows, so they sit with the axis labels
            text_size_pt=text_size_params.axis_label_size,
        )
    for x_pos, num_cells in ((0.025, left_side_resolution), (0.975, right_side_resolution)):
        annotate_panel.add_text(
            panel=panel,
            x_pos_fraction=x_pos,
            y_pos_fraction=0.025,
            label=f"conserves\n{100.0 * energy_conservation_lookup[num_cells]:.1f}\\% / orbit",
            x_alignment=(
                box_positions.Positions.Side.Left if x_pos < 0.5 else box_positions.Positions.Side.Right
            ),
            y_alignment=box_positions.Positions.Side.Bottom,
        )
    ## these sit outside the panel, so they are placed directly rather than through `add_text`
    panel.text(
        0.5,
        1.02,
        "initial profile",
        transform=panel.transAxes,
        ha="center",
        va="bottom",
        fontsize=text_size_params.axis_label_size,
        color=theme_params.foreground_color,
    )
    panel.text(
        0.5,
        -0.02,
        f"profile after {NUM_ORBITS} orbits",
        transform=panel.transAxes,
        ha="center",
        va="top",
        fontsize=text_size_params.axis_label_size,
        color=theme_params.foreground_color,
    )
    palette = add_color.make_palette(
        config=add_color.SequentialConfig(
            palette_name=PALETTE_NAME,
            palette_range=PALETTE_RANGE,
        ),
        value_range=VALUE_RANGE,
    )
    add_color.add_colorbar(
        panels=panel,
        palette=palette,
        label=r"$\log_{10}(b^2 / 2)$",
        colorbar_side="right",
        ## nothing sits between the panel and the bar, so it needs less room than two panels do
        colorbar_gap_pt=panel_gaps.col_pt / 2.0,
    )
    manage_figure.save_figure(
        figure=figure,
        figure_path=FIGURE_PATH,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
