## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

## third-party
import numpy

from numpy.typing import NDArray

## personal
from jormi.ww_data import fit_series
from jormi.ww_io import (
    csv_io,
    manage_io,
    manage_log,
)
from jormi.ww_plots import (
    annotate_axis,
    manage_plots,
    style_plots,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_arrays, validate_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class WaveConfig:

    wave_label: str
    wave_data_dir: Path
    wave_data_file_name: str
    axis_y_range: tuple[float, float]
    fit_x_range: tuple[int, int]

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_ordered_pair(
            self.axis_y_range,
            param_name="axis_y_range",
            strict_ordering=True,
        )
        validate_types.ensure_ordered_pair(
            self.fit_x_range,
            param_name="fit_x_range",
            strict_ordering=True,
        )


@dataclass(frozen=True)
class EMFComputeSchemeStyle:
    color: str
    label: str
    zorder: int


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    marker: str
    marker_size: float
    label: str


@dataclass(frozen=True)
class InterpolationSchemeStyle:
    linestyle: str
    label: str


class EMFComputeScheme(Enum):
    Q26 = EMFComputeSchemeStyle(
        color="gold",
        label="Q26",
        zorder=2,
    )
    B25 = EMFComputeSchemeStyle(
        color="cornflowerblue",
        label="B25",
        zorder=1,
    )
    FS17 = EMFComputeSchemeStyle(
        color="forestgreen",
        label="FS17",
        zorder=1,
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class EMFAveragingScheme(Enum):
    B25 = EMFAveragingSchemeStyle(
        marker="D",
        marker_size=8,
        label="B25",
    )
    LD04 = EMFAveragingSchemeStyle(
        marker="o",
        marker_size=4,
        label="LD04",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class InterpolationScheme(Enum):
    PLM = InterpolationSchemeStyle(
        linestyle=":",
        label="PLM",
    )
    PPM = InterpolationSchemeStyle(
        linestyle="--",
        label="PPM",
    )
    PPM_EP = InterpolationSchemeStyle(
        linestyle="-",
        label="PPM-EP",
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Simulation:
    emf_compute_scheme: EMFComputeScheme
    emf_averaging_scheme: EMFAveragingScheme
    interpolation_scheme: InterpolationScheme

    @property
    def as_tag(
        self,
    ) -> str:
        return (
            f"{self.emf_compute_scheme.as_tag}-"
            f"{self.emf_averaging_scheme.as_tag}-"
            f"{self.interpolation_scheme.as_tag}"
        )


@dataclass(frozen=True)
class ConvergenceSeries:
    sim: Simulation
    ncells: NDArray[numpy.float64]
    cell_size: NDArray[numpy.float64]
    error: NDArray[numpy.float64]

    def __post_init__(
        self,
    ) -> None:
        self._ensure_data_array(self.ncells, param_name="ncells")
        self._ensure_data_array(self.cell_size, param_name="cell_size")
        self._ensure_data_array(self.error, param_name="error")
        validate_arrays.ensure_same_shape(
            array_a=self.ncells,
            array_b=self.cell_size,
            param_name_a="ncells",
            param_name_b="cell_size",
        )
        validate_arrays.ensure_same_shape(
            array_a=self.cell_size,
            array_b=self.error,
            param_name_a="cell_size",
            param_name_b="error",
        )

    @staticmethod
    def _ensure_data_array(
        array: NDArray[numpy.float64],
        *,
        param_name: str,
    ) -> None:
        validate_arrays.ensure_nonempty(array, param_name=param_name)
        validate_arrays.ensure_finite(array, param_name=param_name)
        validate_arrays.ensure_1d(array, param_name=param_name)


##
## === CONSTANTS
##

## inputs and outputs
ROOT_DIR: Path = Path(__file__).parents[3]
DATASET_DIR: Path = ROOT_DIR / "datasets/problems"
FIGURE_PATH: Path = ROOT_DIR / "figures/problems/wave-convergence/grid-aligned-convergence.png"

## per-wave details
WAVE_CONFIGS: tuple[WaveConfig, ...] = (
    WaveConfig(
        wave_label="Alfvén (linear)",
        wave_data_dir=Path("alfven-wave-linear/convergence/ideal/angle=0-nx=1-ny=0-nz=0"),
        wave_data_file_name="alfven_wave_convergence.csv",
        axis_y_range=(-12.5, -6),
        fit_x_range=(16, 512),
    ),
    WaveConfig(
        wave_label="Alfvén (circular)",
        wave_data_dir=Path("alfven-wave-circular/convergence"),
        wave_data_file_name="alfven_wave_circular_convergence.csv",
        axis_y_range=(-13, -6),
        fit_x_range=(16, 2048),
    ),
    WaveConfig(
        wave_label="fast",
        wave_data_dir=Path("fast-wave/convergence/nx=1-ny=0-nz=0"),
        wave_data_file_name="fast_wave_convergence.csv",
        axis_y_range=(-12, -6),
        fit_x_range=(16, 512),
    ),
    WaveConfig(
        wave_label="slow",
        wave_data_dir=Path("slow-wave/convergence/nx=1-ny=0-nz=0"),
        wave_data_file_name="slow_wave_convergence.csv",
        axis_y_range=(-12, -5.5),
        fit_x_range=(16, 512),
    ),
)

##
## === HELPER FUNCTIONS
##


def load_grouped_data_series(
    *,
    wave_config: WaveConfig,
) -> list[ConvergenceSeries]:
    data_dir = DATASET_DIR / wave_config.wave_data_dir
    grouped_data_series: list[ConvergenceSeries] = []
    for emf_compute_scheme in EMFComputeScheme:
        for emf_averaging_scheme in EMFAveragingScheme:
            for interpolation_scheme in InterpolationScheme:
                sim = Simulation(
                    emf_compute_scheme=emf_compute_scheme,
                    emf_averaging_scheme=emf_averaging_scheme,
                    interpolation_scheme=interpolation_scheme,
                )
                data_path = data_dir / sim.as_tag / wave_config.wave_data_file_name
                if not data_path.is_file():
                    manage_log.log_warning(text=f"missing: {data_path}")
                    continue
                data_table = csv_io.read_csv_file_into_dict(
                    data_path,
                    verbose=False,
                )
                grouped_data_series.append(
                    ConvergenceSeries(
                        sim=sim,
                        ncells=numpy.asarray(data_table["nx"]),
                        cell_size=numpy.asarray(data_table["dx"]),
                        error=numpy.asarray(data_table["error"]),
                    ),
                )
    return grouped_data_series


def plot_wave_panel(
    *,
    ax: manage_plots.PlotAxis,
    grouped_data_series: list[ConvergenceSeries],
) -> None:
    for data_series in grouped_data_series:
        emf_compute_scheme_style = data_series.sim.emf_compute_scheme.value
        emf_averaging_scheme_style = data_series.sim.emf_averaging_scheme.value
        interpolation_scheme_style = data_series.sim.interpolation_scheme.value
        ax.plot(
            numpy.log10(data_series.cell_size),
            numpy.log10(data_series.error),
            color=emf_compute_scheme_style.color,
            marker=emf_averaging_scheme_style.marker,
            markersize=emf_averaging_scheme_style.marker_size,
            markerfacecolor="none",
            markeredgecolor=emf_compute_scheme_style.color,
            markeredgewidth=1.5,
            linestyle=interpolation_scheme_style.linestyle,
            linewidth=1.5,
            zorder=emf_compute_scheme_style.zorder,
        )


def overlay_reference_slope(
    *,
    ax: manage_plots.PlotAxis,
    wave_config: WaveConfig,
    grouped_data_series: list[ConvergenceSeries],
) -> None:
    """Overlay a reference slope anchored halfway between the PPM and PPM-EP data series."""
    reference_slope: float = 2.0
    reference_anchor_ncells: int = 128
    reference_sim_ppm = Simulation(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
        interpolation_scheme=InterpolationScheme.PPM,
    )
    reference_sim_ppm_ep = Simulation(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
        interpolation_scheme=InterpolationScheme.PPM_EP,
    )
    data_series_ppm = next(
        data_series for data_series in grouped_data_series if data_series.sim == reference_sim_ppm
    )
    data_series_ppm_ep = next(
        data_series for data_series in grouped_data_series if data_series.sim == reference_sim_ppm_ep
    )
    anchor_index = int(numpy.argmin(numpy.abs(data_series_ppm.ncells - reference_anchor_ncells)))
    x_anchor = numpy.log10(data_series_ppm.cell_size[anchor_index])
    y_ppm = numpy.log10(data_series_ppm.error[anchor_index])
    y_ppm_ep = numpy.log10(data_series_ppm_ep.error[anchor_index])
    y_anchor = 0.5 * (y_ppm + y_ppm_ep)
    intercept = fit_series.get_linear_intercept(
        slope=reference_slope,
        x_ref=x_anchor,
        y_ref=y_anchor,
    )
    min_ncells, max_ncells = wave_config.fit_x_range
    start_index = int(numpy.argmin(numpy.abs(data_series_ppm.ncells - min_ncells)))
    end_index = int(numpy.argmin(numpy.abs(data_series_ppm.ncells - max_ncells)))
    x_values = numpy.log10(data_series_ppm.cell_size[[start_index, end_index]])
    y_values = reference_slope * x_values + intercept
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=y_values,
        color="black",
        linestyle="-.",
        linewidth=1.5,
        alpha=1.0,
        zorder=0.5,
    )


def set_resolution_ticks(
    *,
    ax: manage_plots.PlotAxis,
    ncells: NDArray[numpy.float64],
    cell_sizes: NDArray[numpy.float64],
    show_tick_labels: bool,
    show_axis_label: bool,
) -> None:
    ax.set_xticks(numpy.log10(cell_sizes))
    ax.set_xticklabels([str(int(n)) for n in ncells])
    ax.tick_params(labelbottom=show_tick_labels)
    ax.minorticks_off()
    if show_axis_label:
        ax.set_xlabel("resolution")


def add_delta_x_axis(
    *,
    ax: manage_plots.PlotAxis,
    show_tick_labels: bool,
    show_axis_label: bool,
) -> None:
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.tick_params(labeltop=show_tick_labels)
    if show_axis_label:
        top_ax.set_xlabel(r"$\log_{10} \Delta x$")


def add_emf_compute_scheme_legend(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["o" for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        marker_size=0,  # hide the marker handle, only leave the coloured label text
        text_color="markerfacecolor",
        anchor_point=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
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
        anchor_point=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


def add_interpolation_scheme_legend(
    *,
    ax: manage_plots.PlotAxis,
) -> None:
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[scheme.value.linestyle for scheme in InterpolationScheme],
        labels=[scheme.value.label for scheme in InterpolationScheme],
        colors=["black" for _ in InterpolationScheme],
        line_width=1.2,
        text_color="black",
        anchor_point=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    style_plots.set_theme()
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    fig, axs = manage_plots.create_figure(
        num_rows=len(WAVE_CONFIGS),
        num_cols=1,
        axis_shape=(3.5, 6),
        share_x=True,
        share_y=False,
    )
    for row_index, wave_config in enumerate(WAVE_CONFIGS):
        ax = axs[row_index, 0]
        grouped_data_series = load_grouped_data_series(wave_config=wave_config)
        is_first_row: bool = row_index == 0
        is_last_row: bool = row_index == len(WAVE_CONFIGS) - 1
        set_resolution_ticks(
            ax=ax,
            ncells=grouped_data_series[0].ncells,
            cell_sizes=grouped_data_series[0].cell_size,
            show_tick_labels=is_last_row,
            show_axis_label=is_last_row,
        )
        add_delta_x_axis(
            ax=ax,
            show_tick_labels=is_first_row,
            show_axis_label=is_first_row,
        )
        plot_wave_panel(
            ax=ax,
            grouped_data_series=grouped_data_series,
        )
        overlay_reference_slope(
            ax=ax,
            wave_config=wave_config,
            grouped_data_series=grouped_data_series,
        )
        ax.set_ylim(wave_config.axis_y_range)
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.05,
            y_pos=0.05,
            x_alignment=box_positions.Positions.Side.Left,
            y_alignment=box_positions.Positions.Side.Bottom,
            label=wave_config.wave_label,
        )
    axs[0, 0].invert_xaxis()
    add_emf_compute_scheme_legend(ax=axs[0, 0])
    add_emf_averaging_scheme_legend(ax=axs[1, 0])
    add_interpolation_scheme_legend(ax=axs[2, 0])
    fig.supylabel(r"$\log_{10} \, \mathrm{error}$", x=-0.05)
    manage_plots.save_figure(
        fig=fig,
        fig_path=FIGURE_PATH,
        dpi=200,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
