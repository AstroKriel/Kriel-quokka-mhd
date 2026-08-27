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
    annotate_panel,
    manage_figure,
    style_figure,
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
        label="B25a",
        color="cornflowerblue",
        zorder=1,
    )
    FS17 = EMFComputeSchemeStyle(
        label="FS18",
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
        label="B25b",
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

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems/brio-wu-shock-tube"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/brio-wu-shock-tube/ncells=256/scheme-comparison.png"
DISCONTINUITY_POSITION: float = 0.5

## plotting details
MARKER_PLOT_KWARGS: dict[str, Any] = {
    "markerfacecolor": "none",
    "markersize": 2.5,
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
            label=r"$\rho$",
        ),
        pressure=ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value,
            label=r"$p$",
        ),
        pressure_ratio=ComponentArrays(
            position=pressure_profile.position,
            field_value=pressure_profile.field_value / density_profile.field_value,
            label=r"$p / \rho$",
        ),
        velocity_x0=velocity_profile.components["x_0"],
        velocity_x1=velocity_profile.components["x_1"],
        magnetic_x1=magnetic_profile.components["x_1"],
    )


def load_solution_time(
    *,
    sim_dir: Path,
) -> float:
    """Load the plotted snapshot time from a simulation's density profile."""
    density_path = next((sim_dir / "extracted").glob("density-axis=x_0-index=*.json"))
    return ScalarProfile.load_from_file(file_path=density_path).step_time


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
            label=r"$\rho$",
        ),
        pressure=ComponentArrays(
            position=profiles.pressure.position,
            field_value=compute_moving_average(profiles.pressure.field_value),
            label=r"$p$",
        ),
        pressure_ratio=ComponentArrays(
            position=profiles.pressure_ratio.position,
            field_value=compute_moving_average(profiles.pressure_ratio.field_value),
            label=r"$p / \rho$",
        ),
        velocity_x0=ComponentArrays(
            position=profiles.velocity_x0.position,
            field_value=compute_moving_average(profiles.velocity_x0.field_value),
            label=r"$\left[\vec{v}\right]_0$",
        ),
        velocity_x1=ComponentArrays(
            position=profiles.velocity_x1.position,
            field_value=compute_moving_average(profiles.velocity_x1.field_value),
            label=r"$\left[\vec{v}\right]_1$",
        ),
        magnetic_x1=ComponentArrays(
            position=profiles.magnetic_x1.position,
            field_value=compute_moving_average(profiles.magnetic_x1.field_value),
            label=r"$\left[\vec{b}\right]_1$",
        ),
    )


def shift_profiles(
    *,
    profiles: ShockTubeProfiles,
    shift: float,
) -> ShockTubeProfiles:

    def _shifted(
        component: ComponentArrays,
    ) -> ComponentArrays:
        return ComponentArrays(
            position=component.position - shift,
            field_value=component.field_value,
            label=component.label,
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


def build_axis_bounds(
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> manage_figure.PanelBounds:
    return manage_figure.PanelBounds(
        x_min_fraction=x_lo,
        y_min_fraction=y_lo,
        x_width_fraction=x_hi - x_lo,
        y_width_fraction=y_hi - y_lo,
    )


def add_zoom_inset(
    *,
    panel: manage_figure.Panel,
    axis_ranges: manage_figure.PanelBounds,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: str,
) -> None:
    inset_ax = panel.inset_axes(
        (
            axis_ranges.x_min_fraction,
            axis_ranges.y_min_fraction,
            axis_ranges.x_width_fraction,
            axis_ranges.y_width_fraction,
        ),
    )
    for line in panel.get_lines():
        inset_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            marker=line.get_marker(),
            markeredgecolor=line.get_markeredgecolor(),
            markerfacecolor=line.get_markerfacecolor(),
            markersize=line.get_markersize(),
            markeredgewidth=line.get_markeredgewidth(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            zorder=line.get_zorder(),
        )
    inset_ax.set_xlim(x_range)
    inset_ax.set_ylim(y_range)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(color)
    panel.indicate_inset_zoom(inset_ax, edgecolor=color)


def add_emf_compute_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        marker_first=False,  # put the (invisible) handle after the text, so text hugs the left edge
        anchor_point_fraction=(0.025, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


def add_emf_averaging_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[scheme.value.marker for scheme in EMFAveragingScheme],
        labels=[scheme.value.label for scheme in EMFAveragingScheme],
        colors=["black" for _ in EMFAveragingScheme],
        marker_size_pt=4,
        anchor_point_fraction=(0.0, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )


@dataclass(frozen=True)
class ReferenceSchemeStyle:
    label: str
    color: str


def add_reference_scheme_legend(
    *,
    panel: manage_figure.Panel,
    styles: tuple[ReferenceSchemeStyle, ...],
) -> None:
    handles = [
        mpl_line2d(
            [0],
            [0],
            marker="s",
            linewidth=0,
            markeredgecolor=style.color,
            markerfacecolor="none",
            markeredgewidth=0.3,
            markersize=4,
        ) for style in styles
    ]
    legend = panel.legend(
        handles=handles,
        labels=[style.label for style in styles],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
    )
    panel.add_artist(legend)


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    ## the legends name the curves the same way the panel annotations do, so they read as the
    ## same kind of text, set a quarter point smaller so they stay the quieter of the two
    default_text_sizes = style_figure.TextSizeParams()
    legend_size = default_text_sizes.annotation_size - 0.25
    style_figure.set_figure_params(
        figure_params=style_figure.FigureParams(
            text_size_params=style_figure.TextSizeParams(
                legend_level=default_text_sizes.compute_level_at_size(text_size_pt=legend_size),
            ),
            legend_params=style_figure.LegendParams(
                frame_margin_em=0.0,
                artist_text_gap_em=0.05,
            ),
        ),
    )
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    reference_sim_dir = DATASET_DIR / "ncells=8192/hlld" / get_sim_tag(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
    )
    solution_time = load_solution_time(sim_dir=reference_sim_dir)
    reference_sim_profiles = shift_profiles(
        profiles=compute_smoothed_profiles(
            profiles=load_sim_profiles(sim_dir=reference_sim_dir),
        ),
        shift=DISCONTINUITY_POSITION,
    )
    llf_sim_profiles = shift_profiles(
        profiles=load_sim_profiles(
            sim_dir=DATASET_DIR / "ncells=256/llf" / get_sim_tag(
                emf_compute_scheme=EMFComputeScheme.Q26,
                emf_averaging_scheme=EMFAveragingScheme.B25,
            ),
        ),
        shift=DISCONTINUITY_POSITION,
    )
    figure, panel_grid = manage_figure.create_figure(
        num_panel_cols=2,
        num_panel_rows=3,
        panel_aspect_ratio=1.494,
        ## the rows share an x axis, so only their frames sit in the gap, not tick labels
        panel_row_gap_pt=5.0,
        ## drawn at the width the paper prints it at, so its text is the size it asks for
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
            **MARKER_PLOT_KWARGS,
            "marker": "s",
            "markeredgecolor": "deeppink",
            "zorder": 0,
        },
    )
    ppm_sim_profiles = shift_profiles(
        profiles=load_sim_profiles(
            sim_dir=DATASET_DIR / "ncells=256/hlld/q26-b25-ppm",
        ),
        shift=DISCONTINUITY_POSITION,
    )
    plot_profiles(
        panel_grid=panel_grid,
        profiles=ppm_sim_profiles,
        plot_kwargs={
            **MARKER_PLOT_KWARGS,
            "marker": "s",
            "markeredgecolor": "purple",
            "zorder": 0,
        },
    )
    for emf_compute_scheme in EMFComputeScheme:
        for emf_averaging_scheme in EMFAveragingScheme:
            sim_dir = DATASET_DIR / "ncells=256/hlld" / get_sim_tag(
                emf_compute_scheme=emf_compute_scheme,
                emf_averaging_scheme=emf_averaging_scheme,
            )
            sim_profiles = shift_profiles(
                profiles=load_sim_profiles(sim_dir=sim_dir),
                shift=DISCONTINUITY_POSITION,
            )
            plot_profiles(
                panel_grid=panel_grid,
                profiles=sim_profiles,
                plot_kwargs={
                    **MARKER_PLOT_KWARGS,
                    "marker": emf_averaging_scheme.value.marker,
                    "markeredgecolor": emf_compute_scheme.value.color,
                    "zorder": emf_compute_scheme.value.zorder,
                },
            )
    add_zoom_inset(
        panel=panel_grid[1, 0],
        axis_ranges=build_axis_bounds(
            x_lo=0.7,
            x_hi=0.965,
            y_lo=0.5,
            y_hi=0.95,
        ),
        x_range=(0.125, 0.35),
        y_range=(-0.31, -0.19),
        color="lightgrey",
    )
    add_zoom_inset(
        panel=panel_grid[1, 1],
        axis_ranges=build_axis_bounds(
            x_lo=0.035,
            x_hi=0.4,
            y_lo=0.35,
            y_hi=0.9,
        ),
        x_range=(0.05, 0.15),
        y_range=(1.4, 1.55),
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
    add_reference_scheme_legend(
        panel=panel_grid[1, 0],
        styles=(
            ReferenceSchemeStyle(
                label="PPM-EP + LLF",
                color="deeppink",
            ),
            ReferenceSchemeStyle(
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
