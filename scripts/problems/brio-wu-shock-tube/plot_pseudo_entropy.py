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
class ThermalProfiles:
    pressure_ratio: shock_tubes.ProfileArrays
    pseudo_entropy: shock_tubes.ProfileArrays


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/brio-wu-shock-tube"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/brio-wu-shock-tube/ncells=256/pseudo-entropy.png"

## problem setup
ADIABATIC_INDEX: float = 2.0

##
## === HELPER FUNCTIONS
##


def load_sim_profiles(
    *,
    sim_dir: Path,
) -> ThermalProfiles:
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
    return ThermalProfiles(
        pressure_ratio=shock_tubes.ProfileArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value,
        ),
        pseudo_entropy=shock_tubes.ProfileArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value**ADIABATIC_INDEX,
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
    profiles: ThermalProfiles,
) -> ThermalProfiles:
    return ThermalProfiles(
        pressure_ratio=shock_tubes.ProfileArrays(
            position=profiles.pressure_ratio.position,
            field_value=compute_moving_average(profiles.pressure_ratio.field_value),
        ),
        pseudo_entropy=shock_tubes.ProfileArrays(
            position=profiles.pseudo_entropy.position,
            field_value=compute_moving_average(profiles.pseudo_entropy.field_value),
        ),
    )


def shift_profiles(
    *,
    profiles: ThermalProfiles,
    shift: float,
) -> ThermalProfiles:

    def _shifted(
        component: shock_tubes.ProfileArrays,
    ) -> shock_tubes.ProfileArrays:
        return shock_tubes.ProfileArrays(
            position=component.position - shift,
            field_value=component.field_value,
        )

    return ThermalProfiles(
        pressure_ratio=_shifted(profiles.pressure_ratio),
        pseudo_entropy=_shifted(profiles.pseudo_entropy),
    )


def plot_profiles(
    *,
    panel_grid: manage_figure.PanelGrid,
    profiles: ThermalProfiles,
    plot_kwargs: dict[str, Any],
) -> None:
    panel_grid[0, 0].plot(
        profiles.pressure_ratio.position,
        profiles.pressure_ratio.field_value,
        **plot_kwargs,
    )
    panel_grid[0, 1].plot(
        profiles.pseudo_entropy.position,
        profiles.pseudo_entropy.field_value,
        **plot_kwargs,
    )


def add_emf_compute_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in shock_tubes.EMFComputeScheme],
        labels=[_scheme.value.label for _scheme in shock_tubes.EMFComputeScheme],
        colors=[_scheme.value.color for _scheme in shock_tubes.EMFComputeScheme],
        marker_first=False,
        anchor_point_fraction=(0.025, 0.95),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


def add_emf_averaging_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[_scheme.value.marker for _scheme in shock_tubes.EMFAveragingScheme],
        labels=[_scheme.value.label for _scheme in shock_tubes.EMFAveragingScheme],
        colors=["black" for _ in shock_tubes.EMFAveragingScheme],
        marker_size_pt=4,
        anchor_point_fraction=(0.2, 0.95),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


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
    annotation_size = text_size_params.annotation_size_pt - 1.0
    legend_size = text_size_params.annotation_size_pt - 1.25
    paper_style.setup_plotting_script(
        figure_params=dataclasses.replace(
            paper_style.FIGURE_PARAMS,
            text_size_params=dataclasses.replace(
                paper_style.TEXT_SIZE_PARAMS,
                annotation_level=text_size_params.compute_level_at_size(text_size_pt=annotation_size),
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
        num_panel_rows=1,
        panel_aspect_ratio=3.0 / 2.0,
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
    add_emf_compute_scheme_legend(panel=panel_grid[0, 0])
    add_emf_averaging_scheme_legend(panel=panel_grid[0, 0])
    annotate_panel.add_text(
        panel=panel_grid[0, 0],
        x_pos_fraction=0.95,
        y_pos_fraction=0.925,
        label=rf"$t = {solution_time:.2f}$",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=box_positions.Positions.Side.Top,
    )
    shock_tubes.add_reference_scheme_legend(
        panel=panel_grid[0, 1],
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
        anchor_point_fraction=(0.0, 0.95),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )
    panel_grid[0, 0].set_ylabel(r"$p / \rho$")
    panel_grid[0, 1].set_ylabel(r"$p / \rho^\gamma$")
    panel_grid[0, 0].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
    panel_grid[0, 1].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
    panel_grid[0, 1].tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=False,
        labelright=True,
    )
    panel_grid[0, 1].yaxis.set_label_position("right")
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
