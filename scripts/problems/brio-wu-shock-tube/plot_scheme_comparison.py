## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import numpy

from numpy import typing as numpy_typing

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_io import manage_io
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions

## local
from local_helpers import paper_style, shock_tubes

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ShockTubeProfiles:
    density: shock_tubes.ProfileArrays
    pressure: shock_tubes.ProfileArrays
    pressure_ratio: shock_tubes.ProfileArrays
    velocity_x0: shock_tubes.ProfileArrays
    velocity_x1: shock_tubes.ProfileArrays
    magnetic_x1: shock_tubes.ProfileArrays


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/brio-wu-shock-tube"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/brio-wu-shock-tube/ncells=256/scheme-comparison.png"

## the blank row between the emf compute and emf averaging legend entries is drawn at this fraction of the legend text size
EMF_LEGEND_GAP_TEXT_SCALE: float = 0.65

##
## === HELPER FUNCTIONS
##


def load_sim_profiles(
    *,
    sim_dir: Path,
) -> ShockTubeProfiles:
    extracted_dir = sim_dir / "extracted"
    density_glob = "density-axis=x_0-index=*.json"
    density_paths = sorted(extracted_dir.glob(density_glob))
    if not density_paths:
        raise FileNotFoundError(f"no density profile matching `{density_glob}` found in: {extracted_dir}.")
    density_path = density_paths[0]
    index = density_path.stem.split("index=")[-1]
    density_profile = shock_tubes.load_scalar_profile(
        file_path=extracted_dir / f"density-axis=x_0-index={index}.json",
    )
    pressure_profile = shock_tubes.load_scalar_profile(
        file_path=extracted_dir / f"pressure-axis=x_0-index={index}.json",
    )
    velocity_path = extracted_dir / f"velocity-axis=x_0-index={index}.json"
    magnetic_path = extracted_dir / f"magnetic-axis=x_0-index={index}.json"
    return ShockTubeProfiles(
        density=density_profile,
        pressure=pressure_profile,
        pressure_ratio=shock_tubes.ProfileArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value,
        ),
        velocity_x0=shock_tubes.load_vector_profile_component(
            file_path=velocity_path,
            comp_axis=cartesian_axes.CartesianAxis_3D.X0,
        ),
        velocity_x1=shock_tubes.load_vector_profile_component(
            file_path=velocity_path,
            comp_axis=cartesian_axes.CartesianAxis_3D.X1,
        ),
        magnetic_x1=shock_tubes.load_vector_profile_component(
            file_path=magnetic_path,
            comp_axis=cartesian_axes.CartesianAxis_3D.X1,
        ),
    )


def compute_moving_average(
    values: numpy_typing.NDArray[numpy.floating],
    *,
    window_width: int = 35,
) -> numpy_typing.NDArray[numpy.floating]:
    averaging_kernel = numpy.ones(window_width) / window_width
    pad_width = window_width // 2
    padded_values = numpy.pad(
        array=values,
        pad_width=pad_width,
        mode="edge",
    )
    smoothed_values = numpy.convolve(
        a=padded_values,
        v=averaging_kernel,
        mode="same",
    )
    return smoothed_values[pad_width:pad_width + len(values)]


def compute_smoothed_profiles(
    *,
    profiles: ShockTubeProfiles,
) -> ShockTubeProfiles:
    """
    The 8192-cell reference solution is used as an exact solution, however, it carries small-scale ringing
    near discontinuities. We attempt to remove this with smoothing kernal that has a (default) window small
    enough to only smooth-out the ringing (scales smaller than the window), and leaves the structure of the
    discontinuities, unaffected.
    """
    return ShockTubeProfiles(
        density=shock_tubes.ProfileArrays(
            position=profiles.density.position,
            field_value=compute_moving_average(profiles.density.field_value),
        ),
        pressure=shock_tubes.ProfileArrays(
            position=profiles.pressure.position,
            field_value=compute_moving_average(profiles.pressure.field_value),
        ),
        pressure_ratio=shock_tubes.ProfileArrays(
            position=profiles.pressure_ratio.position,
            field_value=compute_moving_average(profiles.pressure_ratio.field_value),
        ),
        velocity_x0=shock_tubes.ProfileArrays(
            position=profiles.velocity_x0.position,
            field_value=compute_moving_average(profiles.velocity_x0.field_value),
        ),
        velocity_x1=shock_tubes.ProfileArrays(
            position=profiles.velocity_x1.position,
            field_value=compute_moving_average(profiles.velocity_x1.field_value),
        ),
        magnetic_x1=shock_tubes.ProfileArrays(
            position=profiles.magnetic_x1.position,
            field_value=compute_moving_average(profiles.magnetic_x1.field_value),
        ),
    )


def shift_profiles(
    *,
    profiles: ShockTubeProfiles,
    shift: float,
) -> ShockTubeProfiles:

    def _shifted(
        component: shock_tubes.ProfileArrays,
    ) -> shock_tubes.ProfileArrays:
        return shock_tubes.ProfileArrays(
            position=component.position - shift,
            field_value=component.field_value,
        )

    return ShockTubeProfiles(
        density=_shifted(profiles.density),
        pressure=_shifted(profiles.pressure),
        pressure_ratio=_shifted(profiles.pressure_ratio),
        velocity_x0=_shifted(profiles.velocity_x0),
        velocity_x1=_shifted(profiles.velocity_x1),
        magnetic_x1=_shifted(profiles.magnetic_x1),
    )


def plot_profiles(
    *,
    panel_grid: manage_figure.PanelGrid,
    profiles: ShockTubeProfiles,
    plot_kwargs: dict[str, Any],
) -> None:
    panel_grid[0, 0].plot(
        profiles.density.position,
        profiles.density.field_value,
        **plot_kwargs,
    )
    panel_grid[0, 1].plot(
        profiles.pressure.position,
        profiles.pressure.field_value,
        **plot_kwargs,
    )
    panel_grid[1, 0].plot(
        profiles.velocity_x0.position,
        profiles.velocity_x0.field_value,
        **plot_kwargs,
    )
    panel_grid[1, 1].plot(
        profiles.pressure_ratio.position,
        profiles.pressure_ratio.field_value,
        **plot_kwargs,
    )
    panel_grid[2, 0].plot(
        profiles.velocity_x1.position,
        profiles.velocity_x1.field_value,
        **plot_kwargs,
    )
    panel_grid[2, 1].plot(
        profiles.magnetic_x1.position,
        profiles.magnetic_x1.field_value,
        **plot_kwargs,
    )


def add_emf_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    """Key the emf compute schemes (label colour) above the emf averaging schemes (marker), split by a blank row."""
    compute_artists: list[str | None] = [None for _ in shock_tubes.EMFComputeScheme]
    compute_labels = [_scheme.value.label for _scheme in shock_tubes.EMFComputeScheme]
    compute_colors = [_scheme.value.color for _scheme in shock_tubes.EMFComputeScheme]
    averaging_artists: list[str | None] = [_scheme.value.marker for _scheme in shock_tubes.EMFAveragingScheme]
    averaging_labels = [_scheme.value.label for _scheme in shock_tubes.EMFAveragingScheme]
    averaging_colors = ["black" for _ in shock_tubes.EMFAveragingScheme]
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=compute_artists + [None] + averaging_artists,
        labels=compute_labels + [""] + averaging_labels,
        colors=compute_colors + ["black"] + averaging_colors,
        marker_size_pt=4,
        anchor_point_fraction=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )
    legend = panel.artists[-1]
    gap_text = legend.get_texts()[len(compute_labels)]
    gap_text.set_fontsize(EMF_LEGEND_GAP_TEXT_SCALE * gap_text.get_fontsize())


##
## === PROGRAM MAIN
##


def main() -> None:
    discontinuity_position = 0.5
    marker_plot_kwargs: dict[str, Any] = {
        "markerfacecolor": "none",
        "markersize": 2.5,
        "markeredgewidth": 0.3,
        "linestyle": "",
    }
    text_size_params = paper_style.TEXT_SIZE_PARAMS
    legend_size = text_size_params.annotation_size_pt - 0.25
    paper_style.setup_plotting_script(
        figure_params=dataclasses.replace(
            paper_style.FIGURE_PARAMS,
            text_size_params=dataclasses.replace(
                paper_style.TEXT_SIZE_PARAMS,
                legend_level=text_size_params.compute_level_at_size(text_size_pt=legend_size),
            ),
            legend_params=style_figure.LegendParams(
                frame_margin_em=0.0,
                entry_text_gap_em=0.05,
            ),
        ),
    )
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    reference_sim_dir = DATASET_DIR / "num_cells=8192/hlld" / shock_tubes.get_sim_tag(
        emf_compute_tag=shock_tubes.EMFComputeScheme.Q26.as_tag,
        emf_averaging_tag=shock_tubes.EMFAveragingScheme.B25.as_tag,
    )
    solution_time = shock_tubes.load_solution_time(sim_dir=reference_sim_dir)
    reference_sim_profiles = shift_profiles(
        profiles=compute_smoothed_profiles(
            profiles=load_sim_profiles(sim_dir=reference_sim_dir),
        ),
        shift=discontinuity_position,
    )
    llf_sim_profiles = shift_profiles(
        profiles=load_sim_profiles(
            sim_dir=DATASET_DIR / "num_cells=256/llf" / shock_tubes.get_sim_tag(
                emf_compute_tag=shock_tubes.EMFComputeScheme.Q26.as_tag,
                emf_averaging_tag=shock_tubes.EMFAveragingScheme.B25.as_tag,
            ),
        ),
        shift=discontinuity_position,
    )
    figure, panel_grid = manage_figure.create_figure(
        num_panel_cols=2,
        num_panel_rows=3,
        panel_aspect_ratio=3.0 / 2.0,
        panel_row_gap_pt=5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=paper_style.page_width(fraction=0.85),
        ),
        share_x_axis=True,
    )
    plot_profiles(
        panel_grid=panel_grid,
        profiles=reference_sim_profiles,
        plot_kwargs={
            "color": "black",
            "linewidth": 0.9,
            "linestyle": "-",
            "zorder": 4,
        },
    )
    plot_profiles(
        panel_grid=panel_grid,
        profiles=llf_sim_profiles,
        plot_kwargs={
            **marker_plot_kwargs,
            "marker": "s",
            "markeredgecolor": "deeppink",
            "zorder": 0,
        },
    )
    ppm_sim_profiles = shift_profiles(
        profiles=load_sim_profiles(
            sim_dir=DATASET_DIR / "num_cells=256/hlld/q26-b25-ppm",
        ),
        shift=discontinuity_position,
    )
    plot_profiles(
        panel_grid=panel_grid,
        profiles=ppm_sim_profiles,
        plot_kwargs={
            **marker_plot_kwargs,
            "marker": "s",
            "markeredgecolor": "purple",
            "zorder": 0,
        },
    )
    for emf_compute_scheme in shock_tubes.EMFComputeScheme:
        for emf_averaging_scheme in shock_tubes.EMFAveragingScheme:
            sim_dir = DATASET_DIR / "num_cells=256/hlld" / shock_tubes.get_sim_tag(
                emf_compute_tag=emf_compute_scheme.as_tag,
                emf_averaging_tag=emf_averaging_scheme.as_tag,
            )
            sim_profiles = shift_profiles(
                profiles=load_sim_profiles(sim_dir=sim_dir),
                shift=discontinuity_position,
            )
            plot_profiles(
                panel_grid=panel_grid,
                profiles=sim_profiles,
                plot_kwargs={
                    **marker_plot_kwargs,
                    "marker": emf_averaging_scheme.value.marker,
                    "markeredgecolor": emf_compute_scheme.value.color,
                    "zorder": emf_compute_scheme.value.zorder,
                },
            )
    shock_tubes.add_zoom_inset(
        panel=panel_grid[1, 0],
        bounds=shock_tubes.build_axis_bounds(
            x_lo=0.7,
            x_hi=0.965,
            y_lo=0.5,
            y_hi=0.95,
        ),
        x_bounds=(0.125, 0.35),
        y_bounds=(-0.31, -0.19),
    )
    shock_tubes.add_zoom_inset(
        panel=panel_grid[1, 1],
        bounds=shock_tubes.build_axis_bounds(
            x_lo=0.035,
            x_hi=0.4,
            y_lo=0.35,
            y_hi=0.9,
        ),
        x_bounds=(0.05, 0.15),
        y_bounds=(2.1, 2.325),
    )
    add_emf_scheme_legend(panel=panel_grid[0, 0])
    annotate_panel.add_text(
        panel=panel_grid[0, 0],
        x_pos_fraction=0.95,
        y_pos_fraction=0.925,
        label=rf"$t = {solution_time:.2f}$",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Top,
    )
    shock_tubes.add_reference_scheme_legend(
        panel=panel_grid[1, 0],
        styles=(
            shock_tubes.ReferenceSchemeStyle(
                label="PPM-EP + LLF",
                color="deeppink",
            ),
            shock_tubes.ReferenceSchemeStyle(
                label="PPM + HLLD",
                color="purple",
            ),
        ),
    )
    panel_grid[0, 0].set_ylabel(r"$\rho$")
    panel_grid[0, 1].set_ylabel(r"$p$")
    panel_grid[1, 0].set_ylabel(r"$u_0$")
    panel_grid[1, 1].set_ylabel(r"$p / \rho$")
    panel_grid[2, 0].set_ylabel(r"$u_1$")
    panel_grid[2, 1].set_ylabel(r"$b_1$")
    panel_grid[2, 0].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
    panel_grid[2, 1].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
    for panel in panel_grid[:, 1]:
        panel.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
        )
        panel.yaxis.set_label_position("right")
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
