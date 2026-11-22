## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
import numpy

from matplotlib import colors as mpl_colors
from matplotlib import lines as mpl_lines
from numpy import typing as numpy_typing

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_plots import manage_figure
from jormi.ww_types import box_positions
from ww_quokka_sims.sim_io.field_diagnostics import profiles as profile_models

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class ProfileArrays:
    position: numpy_typing.NDArray[numpy.floating]
    field_value: numpy_typing.NDArray[numpy.floating]


@dataclass(frozen=True)
class ReferenceSchemeStyle:
    label: str
    color: str


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


##
## === CONSTANTS
##

## hue, saturation, and value of the inset outline and zoom indicator, each in [0, 1]
INSET_OUTLINE_HSV: tuple[float, float, float] = (0.0, 0.0, 0.75)

##
## === FUNCTIONS
##


def get_sim_tag(
    *,
    emf_compute_tag: str,
    emf_averaging_tag: str,
) -> str:
    return f"{emf_compute_tag}-{emf_averaging_tag}-ppm_ep"


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


def load_solution_time(
    *,
    sim_dir: Path,
) -> float:
    """Load the plotted snapshot time from the density profile of a simulation."""
    extracted_dir = sim_dir / "extracted"
    density_glob = "density-axis=x_0-index=*.json"
    density_paths = sorted(extracted_dir.glob(density_glob))
    if not density_paths:
        raise FileNotFoundError(f"no density profile matching `{density_glob}` found in: {extracted_dir}.")
    return profile_models.ScalarFieldProfile.load_from_file(file_path=density_paths[0]).sim_time


def load_scalar_profile(
    *,
    file_path: Path,
) -> ProfileArrays:
    scalar_profile = profile_models.ScalarFieldProfile.load_from_file(file_path=file_path)
    return ProfileArrays(
        position=scalar_profile.position,
        field_value=scalar_profile.field_value,
    )


def load_vector_profile_component(
    *,
    file_path: Path,
    comp_axis: cartesian_axes.CartesianAxis_3D,
) -> ProfileArrays:
    vector_profile = profile_models.VectorFieldProfile.load_from_file(file_path=file_path)
    return ProfileArrays(
        position=vector_profile.position,
        field_value=vector_profile.components[comp_axis].field_value,
    )


def add_zoom_inset(
    *,
    panel: manage_figure.Panel,
    bounds: manage_figure.PanelBounds,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    outline_color = tuple(mpl_colors.hsv_to_rgb(INSET_OUTLINE_HSV))
    inset_panel = panel.inset_axes(
        (
            bounds.x_min_fraction,
            bounds.y_min_fraction,
            bounds.x_width_fraction,
            bounds.y_width_fraction,
        ),
    )
    for line in panel.get_lines():
        inset_panel.plot(
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
    inset_panel.set_xlim(x_bounds)
    inset_panel.set_ylim(y_bounds)
    inset_panel.set_xticks([])
    inset_panel.set_yticks([])
    inset_indicator = panel.indicate_inset_zoom(
        inset_ax=inset_panel,
        edgecolor=outline_color,
    )
    indicator_line_width = inset_indicator.rectangle.get_linewidth()
    for spine in inset_panel.spines.values():
        spine.set_edgecolor(outline_color)
        spine.set_linewidth(indicator_line_width)


def add_reference_scheme_legend(
    *,
    panel: manage_figure.Panel,
    styles: tuple[ReferenceSchemeStyle, ...],
    anchor_point_fraction: tuple[float, float] = (0.0, 0.0),
    anchor_at_corner: box_positions.Positions.Corner = box_positions.Positions.Corner.BottomLeft,
) -> None:
    handles = [
        mpl_lines.Line2D(
            xdata=[0],
            ydata=[0],
            marker="s",
            linewidth=0,
            markeredgecolor=_style.color,
            markerfacecolor="none",
            markeredgewidth=0.3,
            markersize=4,
        ) for _style in styles
    ]
    legend = panel.legend(
        handles=handles,
        labels=[_style.label for _style in styles],
        loc=anchor_at_corner.value,
        bbox_to_anchor=anchor_point_fraction,
        frameon=False,
    )
    panel.add_artist(legend)


## } MODULE
