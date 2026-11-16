## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import numpy

from numpy import typing as numpy_typing

## personal
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.field_diagnostics import profiles as profile_models

## local
from local_helpers import shock_tubes

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ThermalProfiles:
    pressure_ratio: profile_models.ComponentArrays
    pseudo_entropy: profile_models.ComponentArrays


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
        raise FileNotFoundError(f"no density profile matching `{density_glob}` found in: {extracted_dir}")
    density_path = density_paths[0]
    index = density_path.stem.split("index=")[-1]
    density_profile = profile_models.ScalarProfile.load_from_file(
        file_path=extracted_dir / f"density-axis=x_0-index={index}.json",
    )
    pressure_profile = profile_models.ScalarProfile.load_from_file(
        file_path=extracted_dir / f"pressure-axis=x_0-index={index}.json",
    )
    return ThermalProfiles(
        pressure_ratio=profile_models.ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value,
            label=r"$p / \rho$",
        ),
        pseudo_entropy=profile_models.ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value**ADIABATIC_INDEX,
            label=r"$p / \rho^\gamma$",
        ),
    )


def compute_moving_average(
    values: numpy_typing.NDArray[numpy.floating],
    *,
    window_width: int = 35,
) -> numpy_typing.NDArray[numpy.floating]:
    averaging_kernel = numpy.ones(window_width) / window_width
    pad_width = window_width // 2
    padded_values = numpy.pad(values, pad_width, mode="edge")
    return numpy.convolve(padded_values, averaging_kernel, mode="same")[pad_width:pad_width + len(values)]


def compute_smoothed_profiles(
    *,
    profiles: ThermalProfiles,
) -> ThermalProfiles:
    return ThermalProfiles(
        pressure_ratio=profile_models.ComponentArrays(
            position=profiles.pressure_ratio.position,
            field_value=compute_moving_average(profiles.pressure_ratio.field_value),
            label=r"$p / \rho$",
        ),
        pseudo_entropy=profile_models.ComponentArrays(
            position=profiles.pseudo_entropy.position,
            field_value=compute_moving_average(profiles.pseudo_entropy.field_value),
            label=r"$p / \rho^\gamma$",
        ),
    )


def shift_profiles(
    *,
    profiles: ThermalProfiles,
    shift: float,
) -> ThermalProfiles:

    def _shifted(
        component: profile_models.ComponentArrays,
    ) -> profile_models.ComponentArrays:
        return profile_models.ComponentArrays(
            position=component.position - shift,
            field_value=component.field_value,
            label=component.label,
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
        labels=[scheme.value.label for scheme in shock_tubes.EMFComputeScheme],
        colors=[scheme.value.color for scheme in shock_tubes.EMFComputeScheme],
        marker_first=False,  # put the (invisible) handle after the text, so text hugs the left edge
        anchor_point_fraction=(0.025, 0.95),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


def add_emf_averaging_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[scheme.value.marker for scheme in shock_tubes.EMFAveragingScheme],
        labels=[scheme.value.label for scheme in shock_tubes.EMFAveragingScheme],
        colors=["black" for _ in shock_tubes.EMFAveragingScheme],
        marker_size_pt=4,
        anchor_point_fraction=(0.2, 0.95),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    discontinuity_position = 0.5
    marker_plot_kwargs: dict[str, Any] = {
        "markerfacecolor": "none",
        "markersize": 2.5,
        "markeredgewidth": 0.3,
        "linestyle": "",
    }
    default_text_sizes = style_figure.TextSizeParams()
    annotation_size = default_text_sizes.annotation_size_pt - 1.0
    legend_size = default_text_sizes.annotation_size_pt - 1.25
    style_figure.set_figure_params(
        figure_params=style_figure.FigureParams(
            text_size_params=style_figure.TextSizeParams(
                annotation_level=default_text_sizes.compute_level_at_size(text_size_pt=annotation_size),
                legend_level=default_text_sizes.compute_level_at_size(text_size_pt=legend_size),
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
            figure_width=style_figure.FigureWidth(width_fraction=0.85),
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
