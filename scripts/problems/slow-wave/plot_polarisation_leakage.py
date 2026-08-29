## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import annotate_panel, manage_figure, style_figure
from jormi.ww_types import box_positions

## local
from local_helpers import paper_style

##
## === CONFIGURATION
##

## compares the oblique slow-wave leakage test (k = (1, 2, 3), 45 degrees between k and the
## background field) at low and high resolution, to check whether the b_2 leakage seeded by
## non-grid-aligned reconstruction shrinks with resolution, as expected for a convergent scheme.
NUM_CELLS_LOW = 128
NUM_CELLS_HIGH = 512
SCHEME = "q26-b25-ppm_ep"
PROFILE_AXIS = "x_0"
PRIMARY_COMPONENT = "x_0"
SPURIOUS_COMPONENT = "x_2"
NUM_TIME_SAMPLES = 25

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems/slow-wave/correctness/nx=1-ny=2-nz=3"
FIGURE_PATH = ROOT_DIR / "figures/problems/slow-wave/polarisation-leakage.png"

##
## === HELPER FUNCTIONS
##


def load_energy_time_series(
    *,
    num_cells: int,
    component: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return (time, line-integrated energy) of `component`'s deviation from its mean, per snapshot.

    The domain is periodic, so this uses a plain equal-weight Riemann sum (sum * dx), not
    `numpy.trapezoid`: trapezoidal quadrature halves the weight of the first and last sample,
    which is only correct for an open (non-periodic) interval. On a periodic grid there is no such
    edge, and using trapz here introduced a spurious ~1-2% time-dependent oscillation, since the
    profile's phase shifts snapshot to snapshot -- the plain sum is stable to ~0.2% instead.
    """
    extracted_dir = DATASET_DIR / f"num_cells={num_cells}" / SCHEME / "extracted"
    file_paths = sorted(
        extracted_dir.glob(f"magnetic-axis={PROFILE_AXIS}-index=*.json"),
        key=lambda path: int(path.stem.split("index=")[-1].split("-")[0]),
    )
    times = []
    energies = []
    for file_path in file_paths[1:]:
        data = json_io.read_json_file_into_dict(file_path, verbose=False)
        position = numpy.asarray(data["field_comps"][component]["position"])
        field_value = numpy.asarray(data["field_comps"][component]["field_value"])
        deviation_squared = (field_value - field_value.mean())**2
        cell_size = position[1] - position[0]
        times.append(data["step_time"])
        energies.append(deviation_squared.sum() * cell_size)
    return numpy.asarray(times), numpy.asarray(energies)


def compute_leakage_ratio(
    *,
    num_cells: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return (normalized time, log10 leakage ratio) for one resolution, excluding t=0."""
    times, primary_energy = load_energy_time_series(
        num_cells=num_cells,
        component=PRIMARY_COMPONENT,
    )
    _, spurious_energy = load_energy_time_series(
        num_cells=num_cells,
        component=SPURIOUS_COMPONENT,
    )
    ## the run spans exactly two wave periods (t = 4*pi/omega), so the last recorded time is
    ## twice the period; normalizing by this lets the x-axis read as wave phase, not raw time
    wave_period = times[-1] / 2.0
    normalized_times = times / wave_period
    log10_leakage_ratio = numpy.log10(spurious_energy / primary_energy)
    return normalized_times, log10_leakage_ratio


def subsample_evenly(
    *,
    normalized_times: numpy.ndarray,
    log10_leakage_ratio: numpy.ndarray,
    num_samples: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Down-select to `num_samples` points, evenly spread across the series.

    Uses `linspace` rather than `jormi.ww_lists.sample_list`: that helper's stride is an integer
    floor-division (`(num_elems - 1) // (num_samples - 1)`), which does not generally reach the
    final element, silently dropping the series' tail (e.g. 24 points down to 10 stops at index 18
    of 23). `linspace` anchors both endpoints exactly.
    """
    indices_to_keep = numpy.unique(
        numpy.linspace(
            0,
            len(normalized_times) - 1,
            num_samples,
        ).round().astype(int),
    )
    return normalized_times[indices_to_keep], log10_leakage_ratio[indices_to_keep]


def add_saturation_annotation(
    *,
    panel: manage_figure.Panel,
) -> None:
    """Mark where the wave completes one period, so leakage growth saturates."""
    figure_params = style_figure.get_figure_params()
    artist_params = figure_params.artist_params
    text_size_params = figure_params.text_size_params
    panel.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        zorder=1,
    )
    panel.axvspan(
        0.0,
        1.0,
        color="red",
        alpha=0.1,
        linewidth=0.0,
        zorder=0,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.27,
        y_pos_fraction=0.935,
        label="phase pollution",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Top,
        text_color="red",
    )
    panel.annotate(
        "",
        xytext=(1.0, 0.375),
        xy=(1.5, 0.375),
        xycoords=panel.get_xaxis_transform(),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "blue",
            "linewidth": artist_params.line_width_pt,
            ## the head is sized in points, so tie it to the text it sits beside
            "mutation_scale": text_size_params.annotation_size_pt,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.535,
        y_pos_fraction=0.32,
        label="wave returns to\nalready-polluted\nphases",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Top,
        text_color="blue",
    )


def add_tail_annotation(
    *,
    panel: manage_figure.Panel,
    tail_ave: float,
    tail_std: float,
    y_pos_fraction: float,
    y_alignment: box_positions.Positions.Side,
) -> None:
    panel.axhspan(
        tail_ave - tail_std,
        tail_ave + tail_std,
        color="blue",
        alpha=0.15,
        linewidth=0.0,
        zorder=1,
    )
    panel.axhline(
        tail_ave,
        color="blue",
        linestyle=":",
        zorder=2,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.95,
        y_pos_fraction=y_pos_fraction,
        label=rf"${tail_ave:.2f} \pm {tail_std:.2f}$",
        x_alignment=box_positions.Positions.Side.Right,
        y_alignment=y_alignment,
        text_color="blue",
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
    times_low, ratio_low = compute_leakage_ratio(num_cells=NUM_CELLS_LOW)
    times_high, ratio_high = compute_leakage_ratio(num_cells=NUM_CELLS_HIGH)
    ## compute saturation stats from the full (non-subsampled) series for a robust estimate
    tail_low = ratio_low[len(ratio_low) // 2:]
    tail_high = ratio_high[len(ratio_high) // 2:]
    tail_ave_low, tail_std_low = tail_low.mean(), tail_low.std()
    tail_ave_high, tail_std_high = tail_high.mean(), tail_high.std()
    times_low, ratio_low = subsample_evenly(
        normalized_times=times_low,
        log10_leakage_ratio=ratio_low,
        num_samples=NUM_TIME_SAMPLES,
    )
    times_high, ratio_high = subsample_evenly(
        normalized_times=times_high,
        log10_leakage_ratio=ratio_high,
        num_samples=NUM_TIME_SAMPLES,
    )
    figure, panel = manage_figure.create_figure(
        ## chosen by eye
        panel_aspect_ratio=21.0 / 20.0,
        ## drawn at the width the paper prints it at, so its text is the size it asks for
        figure_layout=style_figure.FigureLayout(
            figure_width=style_figure.FigureWidth(width_fraction=0.5),
        ),
    )
    panel.plot(
        times_low,
        ratio_low,
        marker="o",
        color="black",
        zorder=5,
    )
    panel.plot(
        times_high,
        ratio_high,
        color="black",
        marker="s",
        zorder=5,
    )
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[
            "o",
            "s",
        ],
        labels=[
            f"${NUM_CELLS_LOW}^3$",
            f"${NUM_CELLS_HIGH}^3$",
        ],
        colors=["black", "black"],
        anchor_point_fraction=(0.15, 0.0),
        anchor_at_corner=box_positions.Positions.Corner.BottomLeft,
    )
    add_tail_annotation(
        panel=panel,
        tail_ave=tail_ave_low,
        tail_std=tail_std_low,
        y_pos_fraction=0.925,
        y_alignment=box_positions.Positions.Side.Top,
    )
    add_tail_annotation(
        panel=panel,
        tail_ave=tail_ave_high,
        tail_std=tail_std_high,
        y_pos_fraction=0.615,
        y_alignment=box_positions.Positions.Side.Top,
    )
    add_saturation_annotation(panel=panel)
    panel.set_ylim((-13, -5))
    panel.set_xlabel(r"$t / T$")
    panel.set_ylabel(
        r"$\log_{10}\!\left("
        r"\dfrac{\int (b_2 - \langle b_2 \rangle)^2 \mathrm{d}x_0}"
        r"{\int (b_0 - \langle b_0 \rangle)^2 \mathrm{d}x_0}"
        r"\right)$",
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
