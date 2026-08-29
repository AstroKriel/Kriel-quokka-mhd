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
from aegir import exact_solution, mhd_state
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_types
from ww_quokka_sims.sim_io import profile_models

## local
from local_helpers import shock_tubes

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ShockTubeProfiles:
    density: profile_models.ComponentArrays
    pressure: profile_models.ComponentArrays
    velocity_x0: profile_models.ComponentArrays
    velocity_x1: profile_models.ComponentArrays
    velocity_x2: profile_models.ComponentArrays
    magnetic_x1: profile_models.ComponentArrays
    magnetic_x2: profile_models.ComponentArrays
    total_energy: profile_models.ComponentArrays


@dataclass(frozen=True)
class ProblemSetup:
    """Physical setup parameters mirrored from the quokka simulation's TOML config."""

    discontinuity_position: float
    magnetic_field_normal: float
    gamma: float

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_float(
            param=self.magnetic_field_normal,
            param_name="magnetic_field_normal",
        )
        validate_types.ensure_finite_float(
            param=self.gamma,
            param_name="gamma",
            require_positive=True,
        )
        validate_types.ensure_in_bounds(
            param=self.discontinuity_position,
            min_value=0.0,
            max_value=1.0,
            param_name="discontinuity_position",
        )


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/ryu-jones-2a-shock-tube/num_cells=512"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/ryu-jones-2a-shock-tube/ncells=512/scheme-comparison.png"

PROBLEM_SETUP: ProblemSetup = ProblemSetup(
    discontinuity_position=0.5,
    magnetic_field_normal=0.5641895835477562,
    gamma=5.0 / 3.0,
)

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
        raise FileNotFoundError(f"no density profile matching `{density_glob}` found in: {extracted_dir}")
    density_path = density_paths[0]
    index = density_path.stem.split("index=")[-1]
    density_profile = profile_models.ScalarProfile.load_from_file(
        file_path=extracted_dir / f"density-axis=x_0-index={index}.json",
    )
    pressure_profile = profile_models.ScalarProfile.load_from_file(
        file_path=extracted_dir / f"pressure-axis=x_0-index={index}.json",
    )
    velocity_profile = profile_models.VectorProfile.load_from_file(
        file_path=extracted_dir / f"velocity-axis=x_0-index={index}.json",
    )
    magnetic_profile = profile_models.VectorProfile.load_from_file(
        file_path=extracted_dir / f"magnetic-axis=x_0-index={index}.json",
    )
    total_energy_profile = profile_models.ScalarProfile.load_from_file(
        file_path=extracted_dir / f"total_energy-axis=x_0-index={index}.json",
    )
    return ShockTubeProfiles(
        density=profile_models.ComponentArrays(
            position=density_profile.position,
            field_value=density_profile.field_value,
            label=r"$\rho$",
        ),
        pressure=profile_models.ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value,
            label=r"$p$",
        ),
        velocity_x0=velocity_profile.components["x_0"],
        velocity_x1=velocity_profile.components["x_1"],
        velocity_x2=velocity_profile.components["x_2"],
        magnetic_x1=magnetic_profile.components["x_1"],
        magnetic_x2=magnetic_profile.components["x_2"],
        total_energy=profile_models.ComponentArrays(
            position=total_energy_profile.position,
            field_value=total_energy_profile.field_value,
            label=r"$E_\mathrm{tot}$",
        ),
    )


def compute_exact_profiles() -> ShockTubeProfiles:
    left_state = mhd_state.PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=PROBLEM_SETUP.magnetic_field_normal,
        pressure=0.95,
    )
    right_state = mhd_state.PrimitiveState(
        density=1.0,
        velocity_normal=0.0,
        velocity_transverse_1=0.0,
        velocity_transverse_2=0.0,
        magnetic_field_transverse_1=1.1283791670955125,
        magnetic_field_transverse_2=left_state.magnetic_field_transverse_2,
        pressure=1.0,
    )
    riemann_solution = exact_solution.solve_riemann_problem(
        left_state=left_state,
        right_state=right_state,
        magnetic_field_normal=PROBLEM_SETUP.magnetic_field_normal,
        gamma=PROBLEM_SETUP.gamma,
    )
    example_sim_dir = DATASET_DIR / "hlld" / shock_tubes.get_sim_tag(
        emf_compute_tag=shock_tubes.EMFComputeScheme.Q26.as_tag,
        emf_averaging_tag=shock_tubes.EMFAveragingScheme.B25.as_tag,
    )
    solution_time = shock_tubes.load_solution_time(sim_dir=example_sim_dir)
    sampled_domain: numpy_typing.NDArray[numpy.floating] = numpy.linspace(0.0, 1.0, 2001)
    sampled_solution = exact_solution.sample_snapshot(
        riemann_solution=riemann_solution,
        positions=sampled_domain,
        time=solution_time,
        discontinuity_position=PROBLEM_SETUP.discontinuity_position,
    )
    return ShockTubeProfiles(
        density=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.density for _solution in sampled_solution]),
            label=r"$\rho$",
        ),
        pressure=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.pressure for _solution in sampled_solution]),
            label=r"$p$",
        ),
        velocity_x0=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_normal for _solution in sampled_solution]),
            label=r"$\left[\vec{v}\right]_0$",
        ),
        total_energy=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [
                    mhd_state.compute_total_energy(
                        state=_solution,
                        magnetic_field_normal=PROBLEM_SETUP.magnetic_field_normal,
                        gamma=PROBLEM_SETUP.gamma,
                    ) for _solution in sampled_solution
                ],
            ),
            label=r"$E_\mathrm{tot}$",
        ),
        velocity_x1=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_transverse_1 for _solution in sampled_solution]),
            label=r"$\left[\vec{v}\right]_1$",
        ),
        magnetic_x1=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [_solution.magnetic_field_transverse_1 for _solution in sampled_solution],
            ),
            label=r"$\left[\vec{b}\right]_1$",
        ),
        velocity_x2=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_transverse_2 for _solution in sampled_solution]),
            label=r"$\left[\vec{v}\right]_2$",
        ),
        magnetic_x2=profile_models.ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [_solution.magnetic_field_transverse_2 for _solution in sampled_solution],
            ),
            label=r"$\left[\vec{b}\right]_2$",
        ),
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
        profiles.total_energy.position,
        profiles.total_energy.field_value,
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
    panel_grid[3, 0].plot(
        profiles.velocity_x2.position,
        profiles.velocity_x2.field_value,
        **plot_kwargs,
    )
    panel_grid[3, 1].plot(
        profiles.magnetic_x2.position,
        profiles.magnetic_x2.field_value,
        **plot_kwargs,
    )


def shift_profiles(
    *,
    profiles: ShockTubeProfiles,
    shift: float,
) -> ShockTubeProfiles:

    def _shifted(
        component: profile_models.ComponentArrays,
    ) -> profile_models.ComponentArrays:
        return profile_models.ComponentArrays(
            position=component.position - shift,
            field_value=component.field_value,
            label=component.label,
        )

    return ShockTubeProfiles(
        density=_shifted(profiles.density),
        pressure=_shifted(profiles.pressure),
        velocity_x0=_shifted(profiles.velocity_x0),
        velocity_x1=_shifted(profiles.velocity_x1),
        velocity_x2=_shifted(profiles.velocity_x2),
        magnetic_x1=_shifted(profiles.magnetic_x1),
        magnetic_x2=_shifted(profiles.magnetic_x2),
        total_energy=_shifted(profiles.total_energy),
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
        anchor_point_fraction=(0.04, 0.985),
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
        anchor_point_fraction=(0.0, 0.964),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    marker_plot_kwargs: dict[str, Any] = {
        "markerfacecolor": "none",
        "markersize": 2.5,
        "markeredgewidth": 0.3,
        "linestyle": "",
    }
    default_text_sizes = style_figure.TextSizeParams()
    legend_size = default_text_sizes.annotation_size_pt - 0.25
    style_figure.set_figure_params(
        figure_params=style_figure.FigureParams(
            text_size_params=style_figure.TextSizeParams(
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
    exact_profiles = shift_profiles(
        profiles=compute_exact_profiles(),
        shift=PROBLEM_SETUP.discontinuity_position,
    )
    llf_sim_profiles = shift_profiles(
        profiles=load_sim_profiles(
            sim_dir=DATASET_DIR / "llf" / shock_tubes.get_sim_tag(
                emf_compute_tag=shock_tubes.EMFComputeScheme.Q26.as_tag,
                emf_averaging_tag=shock_tubes.EMFAveragingScheme.B25.as_tag,
            ),
        ),
        shift=PROBLEM_SETUP.discontinuity_position,
    )
    solution_time = shock_tubes.load_solution_time(
        sim_dir=DATASET_DIR / "hlld" / shock_tubes.get_sim_tag(
            emf_compute_tag=shock_tubes.EMFComputeScheme.Q26.as_tag,
            emf_averaging_tag=shock_tubes.EMFAveragingScheme.B25.as_tag,
        ),
    )
    figure, panel_grid = manage_figure.create_figure(
        num_panel_cols=2,
        num_panel_rows=4,
        panel_aspect_ratio=3.0 / 2.0,
        panel_row_gap_pt=5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.85),
        ),
        share_x_axis=True,
    )
    plot_profiles(
        panel_grid=panel_grid,
        profiles=exact_profiles,
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
            sim_dir=DATASET_DIR / "hlld" / "q26-b25-ppm",
        ),
        shift=PROBLEM_SETUP.discontinuity_position,
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
            sim_dir = DATASET_DIR / "hlld" / shock_tubes.get_sim_tag(
                emf_compute_tag=emf_compute_scheme.as_tag,
                emf_averaging_tag=emf_averaging_scheme.as_tag,
            )
            sim_profiles = shift_profiles(
                profiles=load_sim_profiles(sim_dir=sim_dir),
                shift=PROBLEM_SETUP.discontinuity_position,
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
        panel=panel_grid[1, 1],
        bounds=shock_tubes.build_axis_bounds(
            x_lo=0.4,
            x_hi=0.85,
            y_lo=0.05,
            y_hi=0.6,
        ),
        x_bounds=(-0.05, 0.25),
        y_bounds=(4.3, 4.6),
        color="lightgrey",
    )
    shock_tubes.add_zoom_inset(
        panel=panel_grid[3, 0],
        bounds=shock_tubes.build_axis_bounds(
            x_lo=0.05,
            x_hi=0.45,
            y_lo=0.05,
            y_hi=0.75,
        ),
        x_bounds=(0.0, 0.25),
        y_bounds=(0.15, 0.35),
        color="lightgrey",
    )
    add_emf_compute_scheme_legend(panel=panel_grid[0, 0])
    add_emf_averaging_scheme_legend(panel=panel_grid[0, 1])
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
    panel_grid[1, 1].set_ylabel(r"$e_\mathrm{tot}$")
    panel_grid[2, 0].set_ylabel(r"$u_1$")
    panel_grid[2, 1].set_ylabel(r"$b_1$")
    panel_grid[3, 0].set_ylabel(r"$u_2$")
    panel_grid[3, 1].set_ylabel(r"$b_2$")
    panel_grid[3, 0].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
    panel_grid[3, 1].set_xlabel(r"$x_0 - x_\mathrm{shock}$")
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
