## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import math

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_data import fit_series, series_types
from jormi.ww_io import json_io, manage_io, manage_log
from jormi.ww_plots import annotate_panel, manage_figure, style_figure
from jormi.ww_types import box_positions

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class EMFComputeSchemeStyle:
    label: str
    color: str


class EMFComputeScheme(Enum):
    Q26 = EMFComputeSchemeStyle(
        label="Q26",
        color="gold",
    )
    B25 = EMFComputeSchemeStyle(
        label="B25a",
        color="cornflowerblue",
    )
    FS17 = EMFComputeSchemeStyle(
        label="FS18",
        color="forestgreen",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    label: str
    marker: str


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

## inputs and outputs
ROOT_DIR = Path(__file__).parents[3]
DATASET_ROOT = ROOT_DIR / "datasets/problems/small-scale-dynamo"
FIGURE_PATH = ROOT_DIR / "figures/problems/small-scale-dynamo/emf-scheme-growth-rates.png"

NCELLS_VALUES = (32, 64, 128, 256)

MARKER_SIZES = {"o": 4, "D": 6}

EMF_SCHEME_VARIANTS: tuple[tuple[EMFComputeScheme, EMFAveragingScheme], ...] = (
    (EMFComputeScheme.FS17, EMFAveragingScheme.LD04),
    (EMFComputeScheme.FS17, EMFAveragingScheme.B25),
    (EMFComputeScheme.B25, EMFAveragingScheme.LD04),
    (EMFComputeScheme.B25, EMFAveragingScheme.B25),
    (EMFComputeScheme.Q26, EMFAveragingScheme.LD04),
    (EMFComputeScheme.Q26, EMFAveragingScheme.B25),
)

LOG10_EMAG_FIT_BOUNDS = (-8.0, -4.0)

EXPECTED_GROWTH_RATE_EXPONENT = 2.0 / 3.0
ANCHOR_NCELLS = 256

##
## === HELPER FUNCTIONS
##


def load_magnetic_energy(
    *,
    ncells: int,
    sim_label: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    dataset = json_io.read_json_file_into_dict(
        DATASET_ROOT / f"num_cells={ncells}" / f"{sim_label}-ppm_ep" / "magnetic_energy-vi_evolution.json",
        verbose=False,
    )
    return numpy.asarray(dataset["sim_times"]), numpy.asarray(dataset["vi_values"])


def round_to_significant_figures(
    *,
    value: float,
    num_significant_figures: int,
) -> float:
    if value == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    scale = 10.0 ** (num_significant_figures - 1 - magnitude)
    return round(value * scale) / scale


def render_value_with_uncertainty(
    *,
    value: float,
    uncertainty: float,
) -> str:
    """Round `uncertainty` to 1 sig-fig, then round `value` to the same precision."""
    rounded_uncertainty = round_to_significant_figures(value=uncertainty, num_significant_figures=1)
    decimal_places = max(-math.floor(math.log10(rounded_uncertainty)), 0)
    rounded_value = round(value, decimal_places)
    return f"{rounded_value:.{decimal_places}f}\\pm{rounded_uncertainty:.{decimal_places}f}"


def measure_growth_rate(
    *,
    sim_times: numpy.ndarray,
    magnetic_energy: numpy.ndarray,
) -> fit_series.LinearFitSummary:
    log10_emag = numpy.log10(magnetic_energy)
    lower_bound, upper_bound = LOG10_EMAG_FIT_BOUNDS
    in_window = (log10_emag >= lower_bound) & (log10_emag <= upper_bound)
    gaussian_series = series_types.GaussianSeries(
        x_values=sim_times[in_window],
        y_values=numpy.log(magnetic_energy[in_window]),
    )
    return fit_series.fit_linear_model(gaussian_series)


def add_emf_compute_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        marker_first=True,
        anchor_point_fraction=(0.965, 0.35),
        anchor_at_corner=box_positions.Positions.Edge.Right,
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
        anchor_point_fraction=(0.975, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomRight,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    default_text_sizes = style_figure.TextSizeParams()
    style_figure.set_figure_params(
        figure_params=style_figure.FigureParams(
            text_size_params=style_figure.TextSizeParams(
                legend_level=default_text_sizes.compute_level_at_size(
                    text_size_pt=default_text_sizes.annotation_size_pt,
                ),
            ),
            legend_params=style_figure.LegendParams(entry_text_gap_em=0.3),
        ),
    )
    growth_rates = {
        f"{compute_scheme.as_tag}-{averaging_scheme.as_tag}": []
        for compute_scheme, averaging_scheme in EMF_SCHEME_VARIANTS
    }
    growth_rate_sigmas = {
        f"{compute_scheme.as_tag}-{averaging_scheme.as_tag}": []
        for compute_scheme, averaging_scheme in EMF_SCHEME_VARIANTS
    }
    print(f"{'ncells':>6} | {'scheme':10} | {'Gamma':>10} | {'sigma':>8} | {'e-fold time':>11} | n")
    print("-" * 62)
    for ncells in NCELLS_VALUES:
        for compute_scheme, averaging_scheme in EMF_SCHEME_VARIANTS:
            sim_label = f"{compute_scheme.as_tag}-{averaging_scheme.as_tag}"
            legend_label = f"{compute_scheme.as_tag.upper()}+{averaging_scheme.as_tag.upper()}"
            sim_times, magnetic_energy = load_magnetic_energy(ncells=ncells, sim_label=sim_label)
            fit_summary = measure_growth_rate(sim_times=sim_times, magnetic_energy=magnetic_energy)
            growth_rate = fit_summary.slope.value
            growth_rate_sigma = fit_summary.slope.sigma
            e_folding_time = 1.0 / growth_rate
            growth_rates[sim_label].append(growth_rate)
            growth_rate_sigmas[sim_label].append(growth_rate_sigma)
            print(
                f"{ncells:>6} | {legend_label:10} | {growth_rate:>10.4f} | "
                f"{growth_rate_sigma:>8.4f} | {e_folding_time:>11.4f} | {fit_summary.num_points}",
            )
    manage_io.create_directory(directory=FIGURE_PATH.parent, verbose=False)
    fig, axs = manage_figure.create_figure_grid(
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
        panel_aspect_ratio=6.0 / 4.5,
    )
    ax = axs[0, 0]
    for compute_scheme, averaging_scheme in EMF_SCHEME_VARIANTS:
        sim_label = f"{compute_scheme.as_tag}-{averaging_scheme.as_tag}"
        ax.plot(
            NCELLS_VALUES,
            numpy.log10(growth_rates[sim_label]),
            color=compute_scheme.value.color,
            linestyle="None",
            marker=averaging_scheme.value.marker,
            markerfacecolor=compute_scheme.value.color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            # markersize=MARKER_SIZES[averaging_scheme.value.marker],
        )
    anchor_index = NCELLS_VALUES.index(ANCHOR_NCELLS)
    anchor_growth_rates = numpy.asarray(
        [
            growth_rates[f"{compute_scheme.as_tag}-{averaging_scheme.as_tag}"][anchor_index]
            for compute_scheme, averaging_scheme in EMF_SCHEME_VARIANTS
        ],
    )
    anchor_growth_rate = numpy.mean(anchor_growth_rates)
    anchor_growth_rate_std = numpy.std(anchor_growth_rates, ddof=1)
    powerlaw_amplitude = fit_series.get_powerlaw_amplitude(
        exponent=EXPECTED_GROWTH_RATE_EXPONENT,
        x_ref=float(ANCHOR_NCELLS),
        y_ref=float(anchor_growth_rate),
    )
    powerlaw_amplitude_sigma = anchor_growth_rate_std / (float(ANCHOR_NCELLS) ** EXPECTED_GROWTH_RATE_EXPONENT)
    ncells_line = numpy.logspace(1.0, 3.0, 50)
    growth_rate_line = powerlaw_amplitude * ncells_line**EXPECTED_GROWTH_RATE_EXPONENT
    annotate_panel.overlay_curve(
        panel=ax,
        x_values=ncells_line,
        y_values=numpy.log10(growth_rate_line),
        color="black",
        linestyle=":",
        linewidth_pt=1.0,
    )
    x_lim = (10.0**1.4, 10.0**2.5)
    y_lim = (-1.35, 0.2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    label_x_fraction = 0.05
    label_log10_ncells = numpy.log10(x_lim[0]) + label_x_fraction * (numpy.log10(x_lim[1]) - numpy.log10(x_lim[0]))
    label_ncells = float(10.0**label_log10_ncells)
    label_log10_growth_rate = numpy.log10(powerlaw_amplitude * label_ncells**EXPECTED_GROWTH_RATE_EXPONENT)
    label_y_fraction = (label_log10_growth_rate - y_lim[0]) / (y_lim[1] - y_lim[0])
    label_rotation_deg = fit_series.get_line_angle(
        slope=EXPECTED_GROWTH_RATE_EXPONENT,
        domain_bounds=(numpy.log10(x_lim[0]), numpy.log10(x_lim[1]), y_lim[0], y_lim[1]),
        aspect_ratio=6.0 / 4.5,
    )
    annotate_panel.add_text(
        panel=ax,
        x_pos_fraction=label_x_fraction,
        y_pos_fraction=label_y_fraction + 0.0325,
        label="turbulent regulated growth",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Bottom,
        rotate_deg=label_rotation_deg,
    )
    figure_params = style_figure.get_figure_params()
    arrow_ncells = float(numpy.sqrt(NCELLS_VALUES[0] * NCELLS_VALUES[1]))
    arrow_y_start = numpy.log10(powerlaw_amplitude * arrow_ncells**EXPECTED_GROWTH_RATE_EXPONENT)
    ax.annotate(
        "",
        xytext=(arrow_ncells, arrow_y_start),
        xy=(arrow_ncells, arrow_y_start - 0.525),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linewidth": figure_params.artist_params.line_width_pt,
            "mutation_scale": figure_params.text_size_params.annotation_size_pt,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    arrow_x_fraction = (numpy.log10(arrow_ncells) - numpy.log10(x_lim[0])) / (
        numpy.log10(x_lim[1]) - numpy.log10(x_lim[0])
    )
    arrow_mid_y = arrow_y_start - 0.525 / 2.0
    arrow_mid_y_fraction = (arrow_mid_y - y_lim[0]) / (y_lim[1] - y_lim[0])
    annotate_panel.add_text(
        panel=ax,
        x_pos_fraction=arrow_x_fraction + 0.02,
        y_pos_fraction=arrow_mid_y_fraction - 0.01,
        label="viscous regulated\ngrowth",
        x_alignment=box_positions.Positions.Side.Left,
    )
    ax.set_xticks(NCELLS_VALUES)
    ax.set_xticklabels([f"${{{ncells}}}^{3}$" for ncells in NCELLS_VALUES])
    ax.minorticks_on()
    ax.set_xlabel(r"resolution")
    ax.set_ylabel(r"$\log_{10}(E_{\rm mag}\ \mathrm{growth\ rate})$")
    add_emf_compute_scheme_legend(panel=ax)
    add_emf_averaging_scheme_legend(panel=ax)
    powerlaw_amplitude_label = render_value_with_uncertainty(
        value=powerlaw_amplitude,
        uncertainty=powerlaw_amplitude_sigma,
    )
    annotate_panel.add_custom_legend(
        panel=ax,
        artists=[":"],
        labels=[rf"$\log_{{10}}\!\big[({powerlaw_amplitude_label})\,N^{{2/3}}\big]$"],
        colors=["black"],
        anchor_point_fraction=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )
    manage_figure.save_figure(figure=fig, figure_path=FIGURE_PATH)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
