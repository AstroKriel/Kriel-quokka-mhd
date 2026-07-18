## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

## third-party
import numpy

from matplotlib.lines import Line2D as mpl_line2d
from numpy.typing import NDArray

## personal (local)
from aegir import exact_solution, mhd_state
from jormi.ww_plots import (
    annotate_axis,
    manage_plots,
    style_plots,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_types
from ww_quokka_sims.sim_io.profile_models import (
    ComponentArrays,
    ScalarProfile,
    VectorProfile,
)

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class EMFComputeSchemeStyle:
    label: str
    color: str
    zorder: int


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    label: str
    marker: str


class EMFComputeScheme(Enum):
    Q26 = EMFComputeSchemeStyle(
        label="Q26",
        color="gold",
        zorder=3,
    )
    B25 = EMFComputeSchemeStyle(
        label="B25",
        color="cornflowerblue",
        zorder=1,
    )
    FS17 = EMFComputeSchemeStyle(
        label="FS17",
        color="forestgreen",
        zorder=2,
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class EMFAveragingScheme(Enum):
    B25 = EMFAveragingSchemeStyle(
        label="B25",
        marker="D",
    )
    LD04 = EMFAveragingSchemeStyle(
        label="LD04",
        marker="o",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class ShockTubeProfiles:
    density: ComponentArrays
    pressure: ComponentArrays
    velocity_x0: ComponentArrays
    velocity_x1: ComponentArrays
    velocity_x2: ComponentArrays
    magnetic_x1: ComponentArrays
    magnetic_x2: ComponentArrays
    total_energy: ComponentArrays


@dataclass(frozen=True)
class QuokkaSetupParams:
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

SETUP_PARAMS: QuokkaSetupParams = QuokkaSetupParams(
    discontinuity_position=0.5,
    magnetic_field_normal=0.5641895835477562,
    gamma=5.0 / 3.0,
)

ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/ryu-jones-2a-shock-tube/ncells=512"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/ryu-jones-2a-shock-tube/ncells=512/scheme-comparison.png"

MARKER_PLOT_KWARGS: dict[str, Any] = {
    "markerfacecolor": "none",
    "markersize": 6,
    "markeredgewidth": 0.3,
    "linestyle": "",
}

##
## === HELPER FUNCTIONS
##


def get_sim_tag(
    *,
    emf_compute_scheme: EMFComputeScheme,
    emf_averaging_scheme: EMFAveragingScheme,
) -> str:
    return f"{emf_compute_scheme.as_tag}-{emf_averaging_scheme.as_tag}-ppm_ep"


def load_sim_profiles(
    *,
    sim_dir: Path,
) -> ShockTubeProfiles:
    extracted_dir = sim_dir / "extracted"
    density_path = next(extracted_dir.glob("density-axis=x_0-index=*.json"))
    index = density_path.stem.split("index=")[-1]
    density_profile = ScalarProfile.load_from_file(
        file_path=extracted_dir / f"density-axis=x_0-index={index}.json",
    )
    pressure_profile = ScalarProfile.load_from_file(
        file_path=extracted_dir / f"pressure-axis=x_0-index={index}.json",
    )
    velocity_profile = VectorProfile.load_from_file(
        file_path=extracted_dir / f"velocity-axis=x_0-index={index}.json",
    )
    magnetic_profile = VectorProfile.load_from_file(
        file_path=extracted_dir / f"magnetic-axis=x_0-index={index}.json",
    )
    total_energy_profile = ScalarProfile.load_from_file(
        file_path=extracted_dir / f"total_energy-axis=x_0-index={index}.json",
    )
    return ShockTubeProfiles(
        density=ComponentArrays(
            position=density_profile.position,
            field_value=density_profile.field_value,
        ),
        pressure=ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value,
        ),
        velocity_x0=velocity_profile.components["x_0"],
        velocity_x1=velocity_profile.components["x_1"],
        velocity_x2=velocity_profile.components["x_2"],
        magnetic_x1=magnetic_profile.components["x_1"],
        magnetic_x2=magnetic_profile.components["x_2"],
        total_energy=ComponentArrays(
            position=total_energy_profile.position,
            field_value=total_energy_profile.field_value,
        ),
    )


def compute_exact_profiles() -> ShockTubeProfiles:
    left_state = mhd_state.PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=SETUP_PARAMS.magnetic_field_normal,
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
        magnetic_field_normal=SETUP_PARAMS.magnetic_field_normal,
        gamma=SETUP_PARAMS.gamma,
    )
    example_sim_dir = DATASET_DIR / "hlld" / get_sim_tag(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
    )
    solution_time = ScalarProfile.load_from_file(
        next((example_sim_dir / "extracted").glob("density-axis=x_0-index=*.json")),
    ).step_time
    sampled_domain: NDArray[numpy.floating] = numpy.linspace(0.0, 1.0, 2001)
    sampled_solution = exact_solution.sample_snapshot(
        riemann_solution=riemann_solution,
        positions=sampled_domain,
        time=solution_time,
        discontinuity_position=SETUP_PARAMS.discontinuity_position,
    )
    return ShockTubeProfiles(
        density=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.density for _solution in sampled_solution]),
        ),
        pressure=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.pressure for _solution in sampled_solution]),
        ),
        velocity_x0=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_normal for _solution in sampled_solution]),
        ),
        total_energy=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [
                    mhd_state.compute_total_energy(
                        state=_solution,
                        magnetic_field_normal=SETUP_PARAMS.magnetic_field_normal,
                        gamma=SETUP_PARAMS.gamma,
                    ) for _solution in sampled_solution
                ],
            ),
        ),
        velocity_x1=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_transverse_1 for _solution in sampled_solution]),
        ),
        magnetic_x1=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [_solution.magnetic_field_transverse_1 for _solution in sampled_solution],
            ),
        ),
        velocity_x2=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array([_solution.velocity_transverse_2 for _solution in sampled_solution]),
        ),
        magnetic_x2=ComponentArrays(
            position=sampled_domain,
            field_value=numpy.array(
                [_solution.magnetic_field_transverse_2 for _solution in sampled_solution],
            ),
        ),
    )


def plot_profiles(
    *,
    axs: manage_plots.PlotAxesGrid,
    profiles: ShockTubeProfiles,
    plot_kwargs: dict[str, Any],
) -> None:
    axs[0, 0].plot(
        profiles.density.position,
        profiles.density.field_value,
        **plot_kwargs,
    )
    axs[0, 1].plot(
        profiles.pressure.position,
        profiles.pressure.field_value,
        **plot_kwargs,
    )
    axs[1, 0].plot(
        profiles.velocity_x0.position,
        profiles.velocity_x0.field_value,
        **plot_kwargs,
    )
    axs[1, 1].plot(
        profiles.total_energy.position,
        profiles.total_energy.field_value,
        **plot_kwargs,
    )
    axs[2, 0].plot(
        profiles.velocity_x1.position,
        profiles.velocity_x1.field_value,
        **plot_kwargs,
    )
    axs[2, 1].plot(
        profiles.magnetic_x1.position,
        profiles.magnetic_x1.field_value,
        **plot_kwargs,
    )
    axs[3, 0].plot(
        profiles.velocity_x2.position,
        profiles.velocity_x2.field_value,
        **plot_kwargs,
    )
    axs[3, 1].plot(
        profiles.magnetic_x2.position,
        profiles.magnetic_x2.field_value,
        **plot_kwargs,
    )


def add_zoom_inset(
    *,
    ax: manage_plots.PlotAxis,
    bounds: manage_plots.AxisBounds,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    inset_ax = ax.inset_axes((
        bounds.x_min,
        bounds.y_min,
        bounds.x_width,
        bounds.y_width,
    ))
    for line in ax.get_lines():
        inset_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            marker=line.get_marker(),
            markeredgecolor=line.get_markeredgecolor(),
            markerfacecolor=line.get_markerfacecolor(),
            markersize=line.get_markersize(),
            markeredgewidth=line.get_markeredgewidth() * 3.0,
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            zorder=line.get_zorder(),
        )
    inset_ax.set_xlim(x_bounds)
    inset_ax.set_ylim(y_bounds)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("red")
    ax.indicate_inset_zoom(inset_ax, edgecolor="red")


def add_emf_compute_scheme_legend(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["o" for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        marker_size=0,
        text_color="markerfacecolor",
        marker_first=False,  # put the (invisible) handle after the text, so text hugs the left edge
        anchor_point=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


def add_emf_averaging_scheme_legend(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[scheme.value.marker for scheme in EMFAveragingScheme],
        labels=[scheme.value.label for scheme in EMFAveragingScheme],
        colors=["black" for _ in EMFAveragingScheme],
        marker_size=7,
        text_color="black",
        anchor_point=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


def add_llf_legend(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    handle = mpl_line2d(
        [0],
        [0],
        marker="s",
        linewidth=0,
        markeredgecolor="deeppink",
        markerfacecolor="none",
        markeredgewidth=0.3,
        markersize=7,
    )
    legend = ax.legend(
        handles=[handle],
        labels=["LLF"],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=16,
        labelcolor="black",
        frameon=False,
        borderpad=0.45,
        handletextpad=0.5,
    )
    ax.add_artist(legend)


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    exact_profiles = compute_exact_profiles()
    llf_sim_profiles = load_sim_profiles(
        sim_dir=DATASET_DIR / "llf" / get_sim_tag(
            emf_compute_scheme=EMFComputeScheme.Q26,
            emf_averaging_scheme=EMFAveragingScheme.B25,
        ),
    )
    fig, axs = manage_plots.create_figure(
        num_cols=2,
        num_rows=4,
        share_x=True,
    )
    plot_profiles(
        axs=axs,
        profiles=exact_profiles,
        plot_kwargs={
            "color": "black",
            "linewidth": 1.5,
            "linestyle": "-",
            "zorder": 4,
        },
    )
    plot_profiles(
        axs=axs,
        profiles=llf_sim_profiles,
        plot_kwargs={
            **MARKER_PLOT_KWARGS,
            "marker": "s",
            "markeredgecolor": "deeppink",
            "zorder": 0,
        },
    )
    for emf_compute_scheme in EMFComputeScheme:
        for emf_averaging_scheme in EMFAveragingScheme:
            sim_dir = DATASET_DIR / "hlld" / get_sim_tag(
                emf_compute_scheme=emf_compute_scheme,
                emf_averaging_scheme=emf_averaging_scheme,
            )
            sim_profiles = load_sim_profiles(sim_dir=sim_dir)
            plot_profiles(
                axs=axs,
                profiles=sim_profiles,
                plot_kwargs={
                    **MARKER_PLOT_KWARGS,
                    "marker": emf_averaging_scheme.value.marker,
                    "markeredgecolor": emf_compute_scheme.value.color,
                    "zorder": emf_compute_scheme.value.zorder,
                },
            )
    add_zoom_inset(
        ax=axs[1, 1],
        bounds=manage_plots.AxisBounds(
            x_min=0.4,
            y_min=0.05,
            x_width=0.85 - 0.4,
            y_width=0.7 - 0.05,
        ),
        x_bounds=(0.45, 0.75),
        y_bounds=(4.3, 4.6),
    )
    add_zoom_inset(
        ax=axs[3, 0],
        bounds=manage_plots.AxisBounds(
            x_min=0.05,
            y_min=0.05,
            x_width=0.45 - 0.05,
            y_width=0.75 - 0.05,
        ),
        x_bounds=(0.5, 0.75),
        y_bounds=(0.15, 0.35),
    )
    add_emf_compute_scheme_legend(ax=axs[0, 0])
    add_emf_averaging_scheme_legend(ax=axs[0, 1])
    add_llf_legend(ax=axs[1, 0])
    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_0$")
    axs[1, 1].set_ylabel(r"$e_\mathrm{tot}$")
    axs[2, 0].set_ylabel(r"$u_1$")
    axs[2, 1].set_ylabel(r"$b_1$")
    axs[3, 0].set_ylabel(r"$u_2$")
    axs[3, 1].set_ylabel(r"$b_2$")
    axs[3, 0].set_xlabel(r"$x_0$")
    axs[3, 1].set_xlabel(r"$x_0$")
    for ax in axs[:, 1]:
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
        )
        ax.yaxis.set_label_position("right")
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=300,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
