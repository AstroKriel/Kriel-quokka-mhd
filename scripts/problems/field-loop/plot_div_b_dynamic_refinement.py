## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

from matplotlib import collections as mpl_collections
from matplotlib import figure as mpl_figure
from matplotlib import lines as mpl_lines
from matplotlib import patches as mpl_patches
from numpy import typing as numpy_typing

## personal
from jormi.ww_io import json_io
from jormi.ww_io import manage_io
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    color_palettes,
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_types import box_positions

## local
from local_helpers import paper_style

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ProblemSetup:
    initial_position: tuple[float, float]
    direction: tuple[float, float]
    period: float
    radius: float


@dataclass(frozen=True)
class DataSlice:
    divb_2d: numpy_typing.NDArray[numpy.floating]
    sim_time: float
    step_index: int


@dataclass(frozen=True)
class RefinedBox:
    """In-plane cell index bounds (end exclusive) on the finest-level grid."""
    amr_level: int
    x0_index_bounds: tuple[int, int]
    x1_index_bounds: tuple[int, int]


@dataclass(frozen=True)
class SlicedBoxLayout:
    """Refined boxes crossing the slice plane; `fine_grid_resolution` is the in-plane cell count on the `finest_level` grid."""
    finest_level: int
    fine_grid_resolution: tuple[int, int]
    refined_boxes: list[RefinedBox]


@dataclass(frozen=True)
class MergedBoxEdge:
    """Straight piece of one level's box outlines along `along_axis`, sitting at grid line `fixed_index` of the other axis."""
    along_axis: int
    fixed_index: int
    index_bounds: tuple[int, int]


@dataclass(frozen=True)
class PDFData:
    sim_time: float
    bin_centers: numpy_typing.NDArray[numpy.floating]
    log10_densities: numpy_typing.NDArray[numpy.floating]


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/field-loop/dynamic-refinement/num_base_cells=48-max_level=3/q26-b25-ppm_ep" / "extracted"
SLICES_FIGURE_PATH = ROOT_DIR / "figures/problems/field-loop/div-b-dynamic-refinement-slices.png"
PDF_FIGURE_PATH = ROOT_DIR / "figures/problems/field-loop/div-b-dynamic-refinement-pdf.png"

## plotting details
AXIS_BOUNDS: plot_data.AxisRanges = ((-1.5, 1.5), (-1.0, 1.0))
TARGET_TIME_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)
TIME_LABEL_X_POS_FRACTIONS = (0.46, 0.54)
BOX_EDGE_LINESTYLE = "--"
BOX_EDGE_LINEWIDTH_PT = 0.5

##
## === LOAD DATA
##


def load_data_slice(
    file_path: Path,
) -> DataSlice:
    with numpy.load(file_path) as npz:
        return DataSlice(
            divb_2d=npz["sarray_2d"],
            sim_time=float(npz["sim_time"]),
            step_index=int(npz["step_index"]),
        )


def load_pdf_data(
    file_path: Path,
) -> PDFData:
    """Load a single-component PDF JSON; the one non-metadata key holds the binned values."""
    pdf_dict = json_io.read_json_file_into_dict(
        file_path=file_path,
        verbose=False,
    )
    metadata_keys = {"sim_time", "step_index", "use_log10_bins"}
    component_keys = [_key for _key in pdf_dict if _key not in metadata_keys]
    if len(component_keys) != 1:
        raise ValueError(f"expected one PDF component in {file_path}; got {component_keys}.")
    component_dict = pdf_dict[component_keys[0]]
    if pdf_dict["use_log10_bins"]:
        bin_centers_key = "log10_bin_centers"
    else:
        bin_centers_key = "bin_centers"
    return PDFData(
        sim_time=float(pdf_dict["sim_time"]),
        bin_centers=numpy.array(component_dict[bin_centers_key], dtype=float),
        log10_densities=numpy.array(component_dict["log10_density"], dtype=float),
    )


def load_sliced_box_layout(
    file_path: Path,
) -> SlicedBoxLayout:
    """Return every refined box that crosses the x_2 = 0 slice plane, indexed on the finest-level grid."""
    layout_dict = json_io.read_json_file_into_dict(
        file_path=file_path,
        verbose=False,
    )
    refinement_ratio = layout_dict["refinement_ratio"]
    base_resolution = layout_dict["base_resolution"]
    x2_min, x2_max = layout_dict["domain_bounds"][2]
    finest_level = max(_box_dict["amr_level"] for _box_dict in layout_dict["boxes"])
    refined_boxes = []
    for box_dict in layout_dict["boxes"]:
        amr_level = box_dict["amr_level"]
        if amr_level == 0:
            continue
        start_index = box_dict["start_index"]
        end_index = box_dict["end_index"]
        x2_cell_width = (x2_max - x2_min) / (base_resolution[2] * refinement_ratio**amr_level)
        x2_bounds = (x2_min + start_index[2] * x2_cell_width, x2_min + end_index[2] * x2_cell_width)
        box_crosses_midplane = (x2_bounds[0] <= 0.0) and (0.0 < x2_bounds[1])
        if box_crosses_midplane:
            scale_to_finest = refinement_ratio**(finest_level - amr_level)
            refined_box = RefinedBox(
                amr_level=amr_level,
                x0_index_bounds=(start_index[0] * scale_to_finest, end_index[0] * scale_to_finest),
                x1_index_bounds=(start_index[1] * scale_to_finest, end_index[1] * scale_to_finest),
            )
            refined_boxes.append(refined_box)
    fine_grid_resolution = (
        base_resolution[0] * refinement_ratio**finest_level,
        base_resolution[1] * refinement_ratio**finest_level,
    )
    return SlicedBoxLayout(
        finest_level=finest_level,
        fine_grid_resolution=fine_grid_resolution,
        refined_boxes=refined_boxes,
    )


##
## === HELPER FUNCTIONS
##


def compute_symmetric_bounds(
    *,
    fields: list[numpy_typing.NDArray[numpy.floating]],
) -> tuple[float, float]:
    """Return symmetric bounds set by a percentile of the magnitude pooled over every field."""
    pooled_values = numpy.concatenate([numpy.ravel(_field) for _field in fields])
    bound = float(
        numpy.nanpercentile(
            a=numpy.abs(pooled_values),
            q=99.9,
        ),
    )
    return (-bound, bound)


def compute_loop_center_at_time(
    *,
    problem_setup: ProblemSetup,
    sim_time: float,
) -> tuple[float, float]:
    """Return the loop's true advected centre at `sim_time`, wrapped into the periodic domain."""

    def wrap_into_periodic_domain(
        *,
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        span = bounds[1] - bounds[0]
        return (value - bounds[0]) % span + bounds[0]

    raw_x0 = problem_setup.initial_position[0] + problem_setup.direction[0] * sim_time
    raw_x1 = problem_setup.initial_position[1] + problem_setup.direction[1] * sim_time
    return (
        wrap_into_periodic_domain(
            value=raw_x0,
            bounds=AXIS_BOUNDS[0],
        ),
        wrap_into_periodic_domain(
            value=raw_x1,
            bounds=AXIS_BOUNDS[1],
        ),
    )


def merge_index_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the sorted union of the intervals; touching intervals join into one."""
    merged_intervals: list[tuple[int, int]] = []
    for interval_start, interval_end in sorted(intervals):
        interval_joins_previous = (len(merged_intervals) > 0) and (interval_start <= merged_intervals[-1][1])
        if interval_joins_previous:
            previous_start, previous_end = merged_intervals[-1]
            merged_intervals[-1] = (previous_start, max(previous_end, interval_end))
        else:
            merged_intervals.append((interval_start, interval_end))
    return merged_intervals


def subtract_index_intervals(
    *,
    intervals: list[tuple[int, int]],
    removed_intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the parts of `intervals` outside `removed_intervals`, which must be sorted and disjoint."""
    remaining_intervals: list[tuple[int, int]] = []
    for interval_start, interval_end in intervals:
        cursor = interval_start
        for removed_start, removed_end in removed_intervals:
            removed_interval_is_clear = (removed_end <= cursor) or (removed_start >= interval_end)
            if removed_interval_is_clear:
                continue
            if removed_start > cursor:
                remaining_intervals.append((cursor, removed_start))
            cursor = max(cursor, removed_end)
        if cursor < interval_end:
            remaining_intervals.append((cursor, interval_end))
    return remaining_intervals


def collect_box_edge_intervals(
    *,
    box_layout: SlicedBoxLayout,
    amr_level: int,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Group the edges of every box on `amr_level` by (along_axis, fixed_index), dropping edges on the domain boundary."""
    edge_intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for refined_box in box_layout.refined_boxes:
        if refined_box.amr_level != amr_level:
            continue
        box_edges = [
            ((0, refined_box.x1_index_bounds[0]), refined_box.x0_index_bounds),
            ((0, refined_box.x1_index_bounds[1]), refined_box.x0_index_bounds),
            ((1, refined_box.x0_index_bounds[0]), refined_box.x1_index_bounds),
            ((1, refined_box.x0_index_bounds[1]), refined_box.x1_index_bounds),
        ]
        for edge_key, index_bounds in box_edges:
            along_axis, fixed_index = edge_key
            normal_resolution = box_layout.fine_grid_resolution[1 - along_axis]
            edge_is_on_boundary = (fixed_index == 0) or (fixed_index == normal_resolution)
            if edge_is_on_boundary:
                continue
            if edge_key not in edge_intervals:
                edge_intervals[edge_key] = []
            edge_intervals[edge_key].append(index_bounds)
    return edge_intervals


def collect_merged_box_edges(
    *,
    box_layout: SlicedBoxLayout,
) -> dict[int, list[MergedBoxEdge]]:
    """Merge box outlines into unique edges per level; a finer level claims any stretch it shares with a coarser one."""
    claimed_intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}
    merged_box_edges_by_level: dict[int, list[MergedBoxEdge]] = {}
    levels_in_slice = {_refined_box.amr_level for _refined_box in box_layout.refined_boxes}
    for amr_level in sorted(levels_in_slice, reverse=True):
        edge_intervals = collect_box_edge_intervals(
            box_layout=box_layout,
            amr_level=amr_level,
        )
        level_merged_box_edges = []
        for edge_key, intervals in edge_intervals.items():
            merged_intervals = merge_index_intervals(intervals)
            previously_claimed_intervals = claimed_intervals.get(edge_key, [])
            unclaimed_intervals = subtract_index_intervals(
                intervals=merged_intervals,
                removed_intervals=previously_claimed_intervals,
            )
            for index_bounds in unclaimed_intervals:
                merged_box_edge = MergedBoxEdge(
                    along_axis=edge_key[0],
                    fixed_index=edge_key[1],
                    index_bounds=index_bounds,
                )
                level_merged_box_edges.append(merged_box_edge)
            claimed_intervals[edge_key] = merge_index_intervals(
                previously_claimed_intervals + merged_intervals,
            )
        merged_box_edges_by_level[amr_level] = level_merged_box_edges
    return merged_box_edges_by_level


##
## === PANEL FUNCTIONS
##


def draw_merged_box_edges(
    *,
    panel: manage_figure.Panel,
    box_layout: SlicedBoxLayout,
    level_palette: color_palettes.ColorPalette,
) -> None:
    cell_widths = [
        (AXIS_BOUNDS[_axis][1] - AXIS_BOUNDS[_axis][0]) / box_layout.fine_grid_resolution[_axis]
        for _axis in (0, 1)
    ]
    merged_box_edges_by_level = collect_merged_box_edges(box_layout=box_layout)
    for amr_level, merged_box_edges in merged_box_edges_by_level.items():
        line_segments = []
        for merged_box_edge in merged_box_edges:
            along_axis = merged_box_edge.along_axis
            normal_axis = 1 - along_axis
            normal_axis_min = AXIS_BOUNDS[normal_axis][0]
            along_axis_min = AXIS_BOUNDS[along_axis][0]
            fixed_coordinate = normal_axis_min + merged_box_edge.fixed_index * cell_widths[normal_axis]
            start_coordinate = along_axis_min + merged_box_edge.index_bounds[0] * cell_widths[along_axis]
            end_coordinate = along_axis_min + merged_box_edge.index_bounds[1] * cell_widths[along_axis]
            if along_axis == 0:
                line_segment = [(start_coordinate, fixed_coordinate), (end_coordinate, fixed_coordinate)]
            else:
                line_segment = [(fixed_coordinate, start_coordinate), (fixed_coordinate, end_coordinate)]
            line_segments.append(line_segment)
        line_collection = mpl_collections.LineCollection(
            segments=line_segments,
            colors=[level_palette.get_color(amr_level)],
            linestyles=BOX_EDGE_LINESTYLE,
            linewidths=BOX_EDGE_LINEWIDTH_PT,
        )
        panel.add_collection(line_collection, autolim=False)


def plot_slice_panel(
    *,
    panel: manage_figure.Panel,
    problem_setup: ProblemSetup,
    data_slice: DataSlice,
    scaled_divb_2d: numpy_typing.NDArray[numpy.floating],
    box_layout: SlicedBoxLayout,
    time_label_x_pos_fraction: float,
    palette_config: add_color.DivergingPaletteConfig,
    colorbar_bounds: tuple[float, float],
    level_palette: color_palettes.ColorPalette,
) -> None:
    figure_params = style_figure.get_figure_params()
    artist_params = figure_params.artist_params
    text_size_params = figure_params.text_size_params
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=scaled_divb_2d,
        data_format="xy",
        axis_ranges=AXIS_BOUNDS,
        colorbar_range=colorbar_bounds,
        palette_config=palette_config,
        add_colorbar=False,
    )
    draw_merged_box_edges(
        panel=panel,
        box_layout=box_layout,
        level_palette=level_palette,
    )
    loop_center = compute_loop_center_at_time(
        problem_setup=problem_setup,
        sim_time=data_slice.sim_time,
    )
    loop_outline = mpl_patches.Circle(
        xy=loop_center,
        radius=problem_setup.radius,
        fill=False,
        edgecolor="red",
        linestyle="--",
        linewidth=artist_params.line_width_pt,
    )
    panel.add_patch(loop_outline)
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=time_label_x_pos_fraction,
        y_pos_fraction=0.05,
        label=rf"$t / T = {data_slice.sim_time / problem_setup.period:.2f}$",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Bottom,
        text_size_pt=text_size_params.annotation_size_pt - 1.0,
    )
    panel.set_ylabel(r"$x_1$")
    panel.tick_params(labelbottom=False)


def plot_pdf_panel(
    *,
    panel: manage_figure.Panel,
    problem_setup: ProblemSetup,
    pdf_data_series: list[PDFData],
) -> None:
    num_pdf_times = 10
    sampled_indices = numpy.linspace(
        start=0,
        stop=len(pdf_data_series) - 1,
        num=num_pdf_times,
        dtype=int,
    )
    sampled_pdfs = [pdf_data_series[_index] for _index in sampled_indices]
    plotted_pdfs = sampled_pdfs[1:]
    time_palette = add_color.make_palette(
        config=add_color.SequentialPaletteConfig(
            palette_name="cmr.bubblegum_r",
            palette_range=(0.05, 0.85),
        ),
        value_range=(
            plotted_pdfs[0].sim_time / problem_setup.period,
            plotted_pdfs[-1].sim_time / problem_setup.period,
        ),
    )
    for pdf_data in plotted_pdfs:
        pdf_is_finite = numpy.isfinite(pdf_data.log10_densities)
        time_fraction = pdf_data.sim_time / problem_setup.period
        curve_color = time_palette.get_color(time_fraction)
        panel.step(
            pdf_data.bin_centers[pdf_is_finite],
            pdf_data.log10_densities[pdf_is_finite],
            where="mid",
            color=curve_color,
        )
    panel.set_xlabel(r"$y \equiv \log_{10}\!\left(\Delta x\,|\nabla \cdot \vec{b}|\right)$")
    panel.set_ylabel(r"$\log_{10}\!\left(\mathrm{PDF}(y)\right)$")
    panel.set_xlim(-63, -16)
    panel.set_ylim(-3, 0.1)
    add_color.add_inset_colorbar(
        panel=panel,
        palette=time_palette,
        label=r"$t / T$",
        colorbar_corner=box_positions.Positions.Corner.TopLeft,
        colorbar_length_fraction=0.55,
        colorbar_aspect_ratio=10.0,
    )


def add_level_legend(
    *,
    panel: manage_figure.Panel,
    level_palette: color_palettes.ColorPalette,
    refined_levels: list[int],
) -> None:
    text_size_params = style_figure.get_figure_params().text_size_params
    level_handles = [
        mpl_lines.Line2D(
            xdata=[],
            ydata=[],
            color=level_palette.get_color(_amr_level),
            linestyle=BOX_EDGE_LINESTYLE,
            linewidth=BOX_EDGE_LINEWIDTH_PT,
            label=rf"level {_amr_level}",
        ) for _amr_level in refined_levels
    ]
    panel.legend(
        handles=level_handles,
        loc="upper left",
        bbox_to_anchor=(0.07, 1.0),
        frameon=False,
        fontsize=text_size_params.legend_size_pt - 1.0,
    )


##
## === FIGURE FUNCTIONS
##


def plot_slices_figure(
    *,
    problem_setup: ProblemSetup,
    data_slices: list[DataSlice],
    box_layouts: list[SlicedBoxLayout],
) -> mpl_figure.Figure:
    figure_params = style_figure.get_figure_params()
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=len(data_slices),
        num_panel_cols=1,
        panel_aspect_ratio=3.0 / 2.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    palette_config = add_color.DivergingPaletteConfig(
        mid_value=0.0,
        palette_name="cmr.prinsenvlag",
        palette_range=(0.15, 0.85),
    )
    scaled_divb_fields = [_data_slice.divb_2d / 1.0e-16 for _data_slice in data_slices]
    colorbar_bounds = compute_symmetric_bounds(fields=scaled_divb_fields)
    finest_level = max(_box_layout.finest_level for _box_layout in box_layouts)
    refined_levels = list(
        range(
            1,
            finest_level + 1,
        ),
    )
    level_palette = add_color.make_palette(
        config=add_color.SequentialPaletteConfig(
            palette_name="cmr.dusk_r",
            palette_range=(0.25, 0.65),
        ),
        value_range=(refined_levels[0], refined_levels[-1]),
    )
    for row_index, data_slice in enumerate(data_slices):
        plot_slice_panel(
            panel=panel_grid[row_index, 0],
            problem_setup=problem_setup,
            data_slice=data_slice,
            scaled_divb_2d=scaled_divb_fields[row_index],
            box_layout=box_layouts[row_index],
            time_label_x_pos_fraction=TIME_LABEL_X_POS_FRACTIONS[row_index],
            palette_config=palette_config,
            colorbar_bounds=colorbar_bounds,
            level_palette=level_palette,
        )
    divb_palette = add_color.make_palette(
        config=palette_config,
        value_range=colorbar_bounds,
    )
    add_color.add_colorbar(
        panels=panel_grid[0, 0],
        palette=divb_palette,
        label=r"$10^{16} \ (\nabla \cdot \vec{b})$",
        colorbar_side="top",
        colorbar_gap_pt=figure_params.figure_layout.panel_gaps.row_pt / 2.0,
        label_gap_pt=figure_params.frame_params.axis_label_gap_pt * 2.0,
    )
    add_level_legend(
        panel=panel_grid[0, 0],
        level_palette=level_palette,
        refined_levels=refined_levels,
    )
    panel_grid[-1, 0].tick_params(labelbottom=True)
    panel_grid[-1, 0].set_xlabel(r"$x_0$")
    return figure


def plot_pdf_figure(
    *,
    problem_setup: ProblemSetup,
    pdf_data_series: list[PDFData],
) -> mpl_figure.Figure:
    figure, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=1,
        num_panel_cols=1,
        panel_aspect_ratio=3.0 / 2.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    plot_pdf_panel(
        panel=panel_grid[0, 0],
        problem_setup=problem_setup,
        pdf_data_series=pdf_data_series,
    )
    return figure


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    manage_io.create_directory(
        directory=SLICES_FIGURE_PATH.parent,
        verbose=False,
    )
    advection_angle = numpy.arctan2(3.0, 2.0)
    problem_setup = ProblemSetup(
        initial_position=(0.0, 0.0),
        direction=(numpy.sin(advection_angle), numpy.cos(advection_angle)),
        period=(AXIS_BOUNDS[0][1] - AXIS_BOUNDS[0][0]) / numpy.sin(advection_angle),
        radius=0.5,
    )
    pdf_glob = "divb_dimensionless-pdf-index=*.json"
    pdf_paths = sorted(DATASET_DIR.glob(pdf_glob))
    if not pdf_paths:
        raise FileNotFoundError(f"no PDF matching `{pdf_glob}` found in: {DATASET_DIR}")
    pdf_data_series = [load_pdf_data(_path) for _path in pdf_paths]
    slice_glob = "magnetic_divergence-slice=x_2-index=*-native_slice.npz"
    slice_paths = sorted(DATASET_DIR.glob(slice_glob))
    if not slice_paths:
        raise FileNotFoundError(f"no slice matching `{slice_glob}` found in: {DATASET_DIR}")
    data_slice_series = [load_data_slice(_path) for _path in slice_paths]
    data_slices = [
        min(
            data_slice_series,
            key=lambda _slice: abs(_slice.sim_time - _fraction * problem_setup.period),
        ) for _fraction in TARGET_TIME_FRACTIONS
    ]
    box_layouts = [
        load_sliced_box_layout(DATASET_DIR / f"amr_boxes-index={_data_slice.step_index:07d}.json")
        for _data_slice in data_slices
    ]
    slices_figure = plot_slices_figure(
        problem_setup=problem_setup,
        data_slices=data_slices,
        box_layouts=box_layouts,
    )
    manage_figure.save_figure(
        figure=slices_figure,
        figure_path=SLICES_FIGURE_PATH,
    )
    pdf_figure = plot_pdf_figure(
        problem_setup=problem_setup,
        pdf_data_series=pdf_data_series,
    )
    manage_figure.save_figure(
        figure=pdf_figure,
        figure_path=PDF_FIGURE_PATH,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
