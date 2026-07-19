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
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    annotate_axis,
    manage_plots,
    style_plots,
)
from jormi.ww_types import box_positions
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
    pressure_ratio: ComponentArrays
    velocity_x0: ComponentArrays
    velocity_x1: ComponentArrays
    magnetic_x1: ComponentArrays


##
## === CONSTANTS
##

ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/brio-wu-shock-tube"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/brio-wu-shock-tube/ncells=256/scheme-comparison.png"

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
    return ShockTubeProfiles(
        density=ComponentArrays(
            position=density_profile.position,
            field_value=density_profile.field_value,
        ),
        pressure=ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value,
        ),
        pressure_ratio=ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value,
        ),
        velocity_x0=velocity_profile.components["x_0"],
        velocity_x1=velocity_profile.components["x_1"],
        magnetic_x1=magnetic_profile.components["x_1"],
    )


def compute_moving_average(
    values: NDArray[numpy.floating],
    *,
    window_width: int = 35,
) -> NDArray[numpy.floating]:
    averaging_kernel = numpy.ones(window_width) / window_width
    pad_width = window_width // 2
    padded_values = numpy.pad(values, pad_width, mode="edge")
    return numpy.convolve(padded_values, averaging_kernel, mode="same")[pad_width:pad_width + len(values)]


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
        density=ComponentArrays(
            position=profiles.density.position,
            field_value=compute_moving_average(profiles.density.field_value),
        ),
        pressure=ComponentArrays(
            position=profiles.pressure.position,
            field_value=compute_moving_average(profiles.pressure.field_value),
        ),
        pressure_ratio=ComponentArrays(
            position=profiles.pressure_ratio.position,
            field_value=compute_moving_average(profiles.pressure_ratio.field_value),
        ),
        velocity_x0=ComponentArrays(
            position=profiles.velocity_x0.position,
            field_value=compute_moving_average(profiles.velocity_x0.field_value),
        ),
        velocity_x1=ComponentArrays(
            position=profiles.velocity_x1.position,
            field_value=compute_moving_average(profiles.velocity_x1.field_value),
        ),
        magnetic_x1=ComponentArrays(
            position=profiles.magnetic_x1.position,
            field_value=compute_moving_average(profiles.magnetic_x1.field_value),
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
        profiles.pressure_ratio.position,
        profiles.pressure_ratio.field_value,
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


def build_axis_bounds(
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> manage_plots.AxisBounds:
    return manage_plots.AxisBounds(
        x_min=x_lo,
        y_min=y_lo,
        x_width=x_hi - x_lo,
        y_width=y_hi - y_lo,
    )


def add_zoom_inset(
    *,
    ax: manage_plots.PlotAxis,
    axis_bounds: manage_plots.AxisBounds,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    inset_ax = ax.inset_axes(
        (
            axis_bounds.x_min,
            axis_bounds.y_min,
            axis_bounds.x_width,
            axis_bounds.y_width,
        )
    )
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
    inset_ax.set_xlim(x_range)
    inset_ax.set_ylim(y_range)
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
        anchor_point=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
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
        anchor_point=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
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
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    reference_sim_dir = DATASET_DIR / "ncells=8192/hlld" / get_sim_tag(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
    )
    reference_sim_profiles = compute_smoothed_profiles(
        profiles=load_sim_profiles(sim_dir=reference_sim_dir),
    )
    llf_sim_profiles = load_sim_profiles(
        sim_dir=DATASET_DIR / "ncells=256/llf" / get_sim_tag(
            emf_compute_scheme=EMFComputeScheme.Q26,
            emf_averaging_scheme=EMFAveragingScheme.B25,
        ),
    )
    fig, axs = manage_plots.create_figure(
        num_cols=2,
        num_rows=3,
        share_x=True,
    )
    plot_profiles(
        axs=axs,
        profiles=reference_sim_profiles,
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
            sim_dir = DATASET_DIR / "ncells=256/hlld" / get_sim_tag(
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
        ax=axs[1, 0],
        axis_bounds=build_axis_bounds(
            x_lo=0.675,
            x_hi=0.975,
            y_lo=0.425,
            y_hi=0.965,
        ),
        x_range=(0.625, 0.85),
        y_range=(-0.31, -0.19),
    )
    add_zoom_inset(
        ax=axs[1, 1],
        axis_bounds=build_axis_bounds(
            x_lo=0.05,
            x_hi=0.5,
            y_lo=0.3,
            y_hi=0.925,
        ),
        x_range=(0.55, 0.65),
        y_range=(1.4, 1.55),
    )
    add_emf_compute_scheme_legend(ax=axs[0, 0])
    add_emf_averaging_scheme_legend(ax=axs[0, 1])
    add_llf_legend(ax=axs[1, 0])
    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_ylabel(r"$p$")
    axs[1, 0].set_ylabel(r"$u_0$")
    axs[1, 1].set_ylabel(r"$p / \rho$")
    axs[2, 0].set_ylabel(r"$u_1$")
    axs[2, 1].set_ylabel(r"$b_1$")
    axs[2, 0].set_xlabel(r"$x_0$")
    axs[2, 1].set_xlabel(r"$x_0$")
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
