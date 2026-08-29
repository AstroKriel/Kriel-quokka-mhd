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

from numpy import typing as numpy_typing

## personal
from jormi.ww_data import fit_series
from jormi.ww_io import (
    csv_io,
    manage_io,
    manage_log,
)
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_arrays, validate_types

## local
from local_helpers import paper_style

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
    label: str
    color: str
    zorder: int


@dataclass(frozen=True)
class EMFAveragingSchemeStyle:
    label: str
    marker: str
    marker_size: float


@dataclass(frozen=True)
class ReconstructionSchemeStyle:
    label: str
    linestyle: str


class EMFComputeScheme(Enum):
    Q26 = EMFComputeSchemeStyle(
        label="Q26",
        color="gold",
        zorder=2,
    )
    B25 = EMFComputeSchemeStyle(
        label="B25a",
        color="cornflowerblue",
        zorder=1,
    )
    FS17 = EMFComputeSchemeStyle(
        label="FS18",
        color="forestgreen",
        zorder=1,
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
        marker_size=5.0,
    )
    LD04 = EMFAveragingSchemeStyle(
        label="LD04",
        marker="o",
        marker_size=2.5,
    )

    @property
    def as_tag(
        self,
    ) -> str:
        return self.name.lower()


class ReconstructionScheme(Enum):
    PLM = ReconstructionSchemeStyle(
        label="PLM",
        linestyle=":",
    )
    PPM = ReconstructionSchemeStyle(
        label="PPM",
        linestyle="--",
    )
    PPM_EP = ReconstructionSchemeStyle(
        label="PPM-EP",
        linestyle="-",
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
    reconstruction_scheme: ReconstructionScheme

    @property
    def as_tag(
        self,
    ) -> str:
        return (
            f"{self.emf_compute_scheme.as_tag}-"
            f"{self.emf_averaging_scheme.as_tag}-"
            f"{self.reconstruction_scheme.as_tag}"
        )


@dataclass(frozen=True)
class ConvergenceSeries:
    sim: Simulation
    num_cells: numpy_typing.NDArray[numpy.float64]
    cell_size: numpy_typing.NDArray[numpy.float64]
    error: numpy_typing.NDArray[numpy.float64]

    def __post_init__(
        self,
    ) -> None:
        self._ensure_data_array(self.num_cells, param_name="num_cells")
        self._ensure_data_array(self.cell_size, param_name="cell_size")
        self._ensure_data_array(self.error, param_name="error")
        validate_arrays.ensure_same_shape(
            array_a=self.num_cells,
            array_b=self.cell_size,
            param_name_a="num_cells",
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
        array: numpy_typing.NDArray[numpy.float64],
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

WAVE_CONFIGS: tuple[WaveConfig, ...] = (
    WaveConfig(
        wave_label="Alfvén (linear)",
        wave_data_dir=Path("alfven-wave-linear/convergence/ideal/angle=0-nx=1-ny=0-nz=0"),
        wave_data_file_name="alfven_wave_convergence.csv",
        axis_y_range=(-12.5, -6.5),
        fit_x_range=(16, 512),
    ),
    WaveConfig(
        wave_label="Alfvén (circular)",
        wave_data_dir=Path("alfven-wave-circular/convergence"),
        wave_data_file_name="alfven_wave_circular_convergence.csv",
        axis_y_range=(-13.5, -6.5),
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
            for reconstruction_scheme in ReconstructionScheme:
                sim = Simulation(
                    emf_compute_scheme=emf_compute_scheme,
                    emf_averaging_scheme=emf_averaging_scheme,
                    reconstruction_scheme=reconstruction_scheme,
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
                        num_cells=numpy.asarray(data_table["nx"]),
                        cell_size=numpy.asarray(data_table["dx"]),
                        error=numpy.asarray(data_table["error"]),
                    ),
                )
    return grouped_data_series


def plot_wave_panel(
    *,
    panel: manage_figure.Panel,
    grouped_data_series: list[ConvergenceSeries],
) -> None:
    for data_series in grouped_data_series:
        emf_compute_scheme_style = data_series.sim.emf_compute_scheme.value
        emf_averaging_scheme_style = data_series.sim.emf_averaging_scheme.value
        reconstruction_scheme_style = data_series.sim.reconstruction_scheme.value
        panel.plot(
            numpy.log10(data_series.cell_size),
            numpy.log10(data_series.error),
            color=emf_compute_scheme_style.color,
            marker=emf_averaging_scheme_style.marker,
            markersize=emf_averaging_scheme_style.marker_size,
            markerfacecolor="none",
            markeredgecolor=emf_compute_scheme_style.color,
            linestyle=reconstruction_scheme_style.linestyle,
            linewidth=0.9,
            zorder=emf_compute_scheme_style.zorder,
        )


def overlay_reference_slope(
    *,
    panel: manage_figure.Panel,
    wave_config: WaveConfig,
    grouped_data_series: list[ConvergenceSeries],
) -> None:
    """Overlay a reference slope anchored halfway between the PPM and PPM-EP data series."""
    reference_slope: float = 2.0
    reference_anchor_num_cells: int = 128
    reference_sim_ppm = Simulation(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
        reconstruction_scheme=ReconstructionScheme.PPM,
    )
    reference_sim_ppm_ep = Simulation(
        emf_compute_scheme=EMFComputeScheme.Q26,
        emf_averaging_scheme=EMFAveragingScheme.B25,
        reconstruction_scheme=ReconstructionScheme.PPM_EP,
    )
    data_series_ppm = next(
        data_series for data_series in grouped_data_series if data_series.sim == reference_sim_ppm
    )
    data_series_ppm_ep = next(
        data_series for data_series in grouped_data_series if data_series.sim == reference_sim_ppm_ep
    )
    anchor_index = int(numpy.argmin(numpy.abs(data_series_ppm.num_cells - reference_anchor_num_cells)))
    x_anchor = numpy.log10(data_series_ppm.cell_size[anchor_index])
    y_ppm = numpy.log10(data_series_ppm.error[anchor_index])
    y_ppm_ep = numpy.log10(data_series_ppm_ep.error[anchor_index])
    y_anchor = 0.5 * (y_ppm + y_ppm_ep)
    intercept = fit_series.get_linear_intercept(
        slope=reference_slope,
        x_ref=x_anchor,
        y_ref=y_anchor,
    )
    min_num_cells, max_num_cells = wave_config.fit_x_range
    start_index = int(numpy.argmin(numpy.abs(data_series_ppm.num_cells - min_num_cells)))
    end_index = int(numpy.argmin(numpy.abs(data_series_ppm.num_cells - max_num_cells)))
    x_values = numpy.log10(data_series_ppm.cell_size[[start_index, end_index]])
    y_values = reference_slope * x_values + intercept
    annotate_panel.overlay_curve(
        panel=panel,
        x_values=x_values,
        y_values=y_values,
        color="black",
        linestyle="-.",
        linewidth_pt=0.9,
        alpha=1.0,
        zorder=0.5,
    )


def set_resolution_ticks(
    *,
    panel: manage_figure.Panel,
    num_cells: numpy_typing.NDArray[numpy.float64],
    cell_sizes: numpy_typing.NDArray[numpy.float64],
    show_tick_labels: bool,
    show_axis_label: bool,
) -> None:
    panel.set_xticks(numpy.log10(cell_sizes))
    panel.set_xticklabels([str(int(num_cells_value)) for num_cells_value in num_cells])
    panel.tick_params(labelbottom=show_tick_labels)
    panel.minorticks_off()
    if show_axis_label:
        panel.set_xlabel("resolution", labelpad=10.0)


def add_delta_x_axis(
    *,
    panel: manage_figure.Panel,
    show_tick_labels: bool,
    show_axis_label: bool,
) -> None:
    top_ax = panel.twiny()
    top_ax.set_xlim(panel.get_xlim())
    top_ax.tick_params(labeltop=show_tick_labels)
    if show_axis_label:
        top_ax.set_xlabel(r"$\log_{10} (\Delta x / L)$", labelpad=10.0)


def add_emf_compute_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in EMFComputeScheme],
        labels=[scheme.value.label for scheme in EMFComputeScheme],
        colors=[scheme.value.color for scheme in EMFComputeScheme],
        anchor_point_fraction=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
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
        text_color="black",
        anchor_point_fraction=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


def add_reconstruction_scheme_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[scheme.value.linestyle for scheme in ReconstructionScheme],
        labels=[scheme.value.label for scheme in ReconstructionScheme],
        colors=["black" for _ in ReconstructionScheme],
        text_color="black",
        anchor_point_fraction=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    paper_style.setup_plotting_script()
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    figure, panel_grid = manage_figure.create_figure(
        num_panel_rows=len(WAVE_CONFIGS),
        num_panel_cols=1,
        panel_aspect_ratio=17.0 / 10.0,
        panel_row_gap_pt=5.0,
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
        share_x_axis=True,
        share_y_axis=False,
    )
    panel_grid[0, 0].invert_xaxis()
    for row_index, wave_config in enumerate(WAVE_CONFIGS):
        panel = panel_grid[row_index, 0]
        grouped_data_series = load_grouped_data_series(wave_config=wave_config)
        is_first_row: bool = row_index == 0
        is_last_row: bool = row_index == len(WAVE_CONFIGS) - 1
        set_resolution_ticks(
            panel=panel,
            num_cells=grouped_data_series[0].num_cells,
            cell_sizes=grouped_data_series[0].cell_size,
            show_tick_labels=is_last_row,
            show_axis_label=is_last_row,
        )
        plot_wave_panel(
            panel=panel,
            grouped_data_series=grouped_data_series,
        )
        overlay_reference_slope(
            panel=panel,
            wave_config=wave_config,
            grouped_data_series=grouped_data_series,
        )
        panel.set_ylim(wave_config.axis_y_range)
        add_delta_x_axis(
            panel=panel,
            show_tick_labels=is_first_row,
            show_axis_label=is_first_row,
        )
        annotate_panel.add_text(
            panel=panel,
            x_pos_fraction=0.05,
            y_pos_fraction=0.05,
            x_alignment=box_positions.Positions.Side.Left,
            y_alignment=box_positions.Positions.Side.Bottom,
            label=wave_config.wave_label,
        )
    add_emf_compute_scheme_legend(panel=panel_grid[0, 0])
    add_emf_averaging_scheme_legend(panel=panel_grid[1, 0])
    add_reconstruction_scheme_legend(panel=panel_grid[2, 0])
    annotate_panel.add_shared_axis_label(
        panels=panel_grid,
        label=r"$\log_{10} (\mbox{relative error})$",
        side=box_positions.Positions.Side.Left,
    )
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
