## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
from matplotlib import lines as mpl_lines

## personal
from jormi.ww_plots import manage_figure
from ww_quokka_sims.sim_io.field_diagnostics import profiles as profile_models

##
## === DATA STRUCTURES
##


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
    """Load the plotted snapshot time from a simulation's density profile."""
    extracted_dir = sim_dir / "extracted"
    density_glob = "density-axis=x_0-index=*.json"
    density_paths = sorted(extracted_dir.glob(density_glob))
    if not density_paths:
        raise FileNotFoundError(f"no density profile matching `{density_glob}` found in: {extracted_dir}")
    return profile_models.ScalarProfile.load_from_file(file_path=density_paths[0]).step_time


def add_zoom_inset(
    *,
    panel: manage_figure.Panel,
    bounds: manage_figure.PanelBounds,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    color: str,
) -> None:
    inset_ax = panel.inset_axes(
        (
            bounds.x_min_fraction,
            bounds.y_min_fraction,
            bounds.x_width_fraction,
            bounds.y_width_fraction,
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
    inset_ax.set_xlim(x_bounds)
    inset_ax.set_ylim(y_bounds)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(color)
    panel.indicate_inset_zoom(inset_ax, edgecolor=color)


def add_reference_scheme_legend(
    *,
    panel: manage_figure.Panel,
    styles: tuple[ReferenceSchemeStyle, ...],
) -> None:
    handles = [
        mpl_lines.Line2D(
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


## } MODULE
