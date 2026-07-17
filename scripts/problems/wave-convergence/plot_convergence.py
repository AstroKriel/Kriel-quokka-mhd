## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_data import fit_series
from jormi.ww_io import csv_io, manage_io
from jormi.ww_plots import annotate_axis, manage_plots, style_plots
from jormi.ww_types import box_positions

##
## === CONFIGURATION
##

## one row per linear MHD wave family; each reads the grid-aligned convergence run.
## `sub_path` is the resolution-ladder subdirectory beneath convergence/.
WAVES = (
    {
        "label": "Alfvén (linear)",
        "problem_dir": "alfven-wave-linear",
        "sub_path": "ideal/angle=0-nx=1-ny=0-nz=0",
        "csv_name": "alfven_wave_convergence.csv",
        "y_lim": (-6, -12.5),
        "reference_max_ncells": 512,
    },
    {
        ## no angle/mode axis for this test, hence the flat sub_path.
        "label": "Alfvén (circular)",
        "problem_dir": "alfven-wave-circular",
        "sub_path": "",
        "csv_name": "alfven_wave_circular_convergence.csv",
        "y_lim": (-6, -13),
        "reference_max_ncells": None,  ## show across the full resolution ladder
    },
    {
        "label": "fast",
        "problem_dir": "fast-wave",
        "sub_path": "nx=1-ny=0-nz=0",
        "csv_name": "fast_wave_convergence.csv",
        "y_lim": (-6, -12),
        "reference_max_ncells": 512,
    },
    {
        "label": "slow",
        "problem_dir": "slow-wave",
        "sub_path": "nx=1-ny=0-nz=0",
        "csv_name": "slow_wave_convergence.csv",
        "y_lim": (-5.5, -12),
        "reference_max_ncells": 512,
    },
)

## scheme tokens, matching the dataset directory names: <reconstruction>-<averaging>-<interpolation>.
EMF_RECONSTRUCTIONS = ("q26", "b25", "fs17")
EMF_AVERAGINGS = ("b25", "ld04")
INTERPOLATIONS = ("plm", "ppm", "ppm_ep")  ## encoded by linestyle; (omitted: "pcm")

## same convention as `brio-wu-shock-tube/plot_scheme_comparison.py`: colour by EMF reconstruction,
## marker by EMF averaging. Linestyle-by-interpolation is specific to this plot, since the wave
## tests (unlike the shock tubes) sweep interpolation order too.
COMPUTE_COLORS = {"q26": "gold", "b25": "cornflowerblue", "fs17": "forestgreen"}
AVERAGING_MARKERS = {"b25": "D", "ld04": "o"}
## the diamond marker reads visually smaller than the circle at equal markersize, so bump it up
MARKERSIZE_BY_AVERAGING = {"b25": 8, "ld04": 4}
## q26 is the recommended reconstruction, so draw it last/on top when combos overlap
ZORDER_BY_RECONSTRUCTION = {"fs17": 1, "b25": 1, "q26": 2}
LINESTYLE_BY_INTERPOLATION = {
    "plm": ":",
    "ppm": "--",
    "ppm_ep": "-",
}

## display shorthand for legend labels, distinct from the lowercase dataset-directory tokens
COMPUTE_LABELS = {"q26": "Q26", "fs17": "FS17", "b25": "B25"}
AVERAGING_LABELS = {"ld04": "LD04", "b25": "B25"}
INTERPOLATION_LABELS = {"plm": "PLM", "ppm": "PPM", "ppm_ep": "PPM-EP"}

## the raw error is plotted (uncompensated), so the slope is the convergence order directly;
## both axes plot the log10 of the data (linear axes), rather than log-scaling the axes
FIELD_LABEL_X = r"$\log_{10} \Delta x$"
FIELD_LABEL_Y = r"$\log_{10} \, \mathrm{error}$"

ROOT_DIR = Path(__file__).parents[3]
DATASET_DIR = ROOT_DIR / "datasets/problems"
FIGURE_PATH = ROOT_DIR / "figures/problems/wave-convergence/wave-convergence.png"

##
## === HELPER FUNCTIONS
##


def get_convergence_dir(
    *,
    wave: dict[str, str],
) -> Path:
    """Path to a wave's resolution-ladder directory."""
    return DATASET_DIR / wave["problem_dir"] / "convergence" / wave["sub_path"]


def load_wave_convergence(
    *,
    wave: dict[str, str],
) -> dict[tuple[str, str, str], tuple[numpy.ndarray, numpy.ndarray]]:
    """Load (dx, error) for every plotted scheme combo of a single wave family.

    Missing combos (eg. a still-running regeneration job) are skipped with a warning
    rather than raising, so the figure can be previewed before every job has finished.
    """
    series: dict[tuple[str, str, str], tuple[numpy.ndarray, numpy.ndarray]] = {}
    convergence_dir = get_convergence_dir(wave=wave)
    for reconstruction in EMF_RECONSTRUCTIONS:
        for averaging in EMF_AVERAGINGS:
            for interpolation in INTERPOLATIONS:
                csv_path = convergence_dir / f"{reconstruction}-{averaging}-{interpolation}" / wave["csv_name"]
                if not csv_path.is_file():
                    print(f"[skip] missing: {csv_path}")
                    continue
                table = csv_io.read_csv_file_into_dict(
                    csv_path,
                    verbose=False,
                )
                cell_size = numpy.asarray(table["dx"])
                error = numpy.asarray(table["error"])
                series[(reconstruction, averaging, interpolation)] = (cell_size, error)
    return series


def plot_wave_panel(
    *,
    ax,
    series: dict[tuple[str, str, str], tuple[numpy.ndarray, numpy.ndarray]],
) -> None:
    """Plot every scheme combo for one wave family onto `ax`, as error vs dx."""
    for reconstruction in EMF_RECONSTRUCTIONS:
        for averaging in EMF_AVERAGINGS:
            for interpolation in INTERPOLATIONS:
                key = (reconstruction, averaging, interpolation)
                if key not in series:
                    continue
                cell_size, error = series[key]
                ax.plot(
                    numpy.log10(cell_size),
                    numpy.log10(error),
                    color=COMPUTE_COLORS[reconstruction],
                    marker=AVERAGING_MARKERS[averaging],
                    markersize=MARKERSIZE_BY_AVERAGING[averaging],
                    markerfacecolor="none",
                    markeredgecolor=COMPUTE_COLORS[reconstruction],
                    markeredgewidth=1.5,
                    linestyle=LINESTYLE_BY_INTERPOLATION[interpolation],
                    linewidth=1.5,
                    zorder=ZORDER_BY_RECONSTRUCTION[reconstruction],
                )


## the reference line's anchor point is set from the recommended scheme (Q26 + Balsara2025),
## halfway between the PPM and PPM-EP interpolation curves, at this resolution
REFERENCE_SLOPE = 2.0
REFERENCE_ANCHOR_NCELLS = 128
REFERENCE_RECONSTRUCTION = "q26"
REFERENCE_AVERAGING = "b25"


def add_reference_slope(
    *,
    ax,
    wave: dict[str, str],
) -> None:
    """Overlay a slope=`REFERENCE_SLOPE` line, anchored halfway (in log-log space) between the
    PPM and PPM-EP curves of the recommended scheme, at `REFERENCE_ANCHOR_NCELLS`.
    """
    convergence_dir = get_convergence_dir(wave=wave)
    table_ppm = csv_io.read_csv_file_into_dict(
        convergence_dir / f"{REFERENCE_RECONSTRUCTION}-{REFERENCE_AVERAGING}-ppm" / wave["csv_name"],
        verbose=False,
    )
    table_ppm_ep = csv_io.read_csv_file_into_dict(
        convergence_dir / f"{REFERENCE_RECONSTRUCTION}-{REFERENCE_AVERAGING}-ppm_ep" / wave["csv_name"],
        verbose=False,
    )
    nx_array = numpy.asarray(table_ppm["nx"])
    dx_array = numpy.asarray(table_ppm["dx"])
    anchor_index = int(numpy.argmin(numpy.abs(nx_array - REFERENCE_ANCHOR_NCELLS)))
    x_ref = numpy.log10(dx_array[anchor_index])
    y_ppm = numpy.log10(numpy.asarray(table_ppm["error"])[anchor_index])
    y_ppm_ep = numpy.log10(numpy.asarray(table_ppm_ep["error"])[anchor_index])
    y_ref = 0.5 * (y_ppm + y_ppm_ep)
    intercept = fit_series.get_linear_intercept(
        slope=REFERENCE_SLOPE,
        x_ref=x_ref,
        y_ref=y_ref,
    )
    ## the line spans from the coarsest resolution up to `reference_max_ncells` (or the finest,
    ## when unset), so it can be truncated per-wave where the data no longer tracks 2nd order
    max_ncells = wave.get("reference_max_ncells") or nx_array[-1]
    end_index = int(numpy.argmin(numpy.abs(nx_array - max_ncells)))
    x_values = numpy.log10(dx_array[[0, end_index]])
    y_values = REFERENCE_SLOPE * x_values + intercept
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


def read_resolution_ladder(
    *,
    wave: dict[str, str],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Read the shared (ncells, dx) ladder for a wave from a representative combo's CSV."""
    combo = f"{EMF_RECONSTRUCTIONS[0]}-{EMF_AVERAGINGS[0]}-{INTERPOLATIONS[0]}"
    csv_path = get_convergence_dir(wave=wave) / combo / wave["csv_name"]
    table = csv_io.read_csv_file_into_dict(
        csv_path,
        verbose=False,
    )
    return numpy.asarray(table["nx"]), numpy.asarray(table["dx"])


def set_resolution_ticks(
    *,
    ax,
    ncells: numpy.ndarray,
    cell_sizes: numpy.ndarray,
) -> None:
    """Set the bottom x-axis ticks to the linear cell count at each log10(dx) tick position."""
    ax.set_xticks(numpy.log10(cell_sizes))
    ax.set_xticklabels([str(int(n)) for n in ncells])
    ax.minorticks_off()
    ax.set_xlabel("resolution")


def add_delta_x_axis(
    *,
    ax,
) -> None:
    """Annotate a top x-axis with log10(dx), using default numeric ticks."""
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())  ## share the (inverted) log10(dx) range of the bottom axis
    top_ax.set_xlabel(FIELD_LABEL_X)


def add_compute_legend(
    *,
    ax,
) -> None:
    """Legend for EMF compute, shown as coloured shorthand text (no marker, since colour alone
    already encodes compute in the data).
    """
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=["o" for _ in EMF_RECONSTRUCTIONS],
        labels=[COMPUTE_LABELS[reconstruction] for reconstruction in EMF_RECONSTRUCTIONS],
        colors=[COMPUTE_COLORS[reconstruction] for reconstruction in EMF_RECONSTRUCTIONS],
        marker_size=0,  ## hide the marker handle, leaving only the coloured label text
        text_color="markerfacecolor",
        anchor_point=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


def add_averaging_legend(
    *,
    ax,
) -> None:
    """Legend for EMF averaging, shown as marker + shorthand."""
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[AVERAGING_MARKERS[averaging] for averaging in EMF_AVERAGINGS],
        labels=[AVERAGING_LABELS[averaging] for averaging in EMF_AVERAGINGS],
        colors=["black" for _ in EMF_AVERAGINGS],
        marker_size=7,
        text_color="black",
        anchor_point=(1.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopRight,
    )


def add_interpolation_legend(
    *,
    ax,
) -> None:
    """Legend for interpolation order, shown as line style + shorthand."""
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[LINESTYLE_BY_INTERPOLATION[interpolation] for interpolation in INTERPOLATIONS],
        labels=[INTERPOLATION_LABELS[interpolation] for interpolation in INTERPOLATIONS],
        colors=["black" for _ in INTERPOLATIONS],
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
    manage_io.create_directory(
        directory=FIGURE_PATH.parent,
        verbose=False,
    )
    fig, axs = manage_plots.create_figure_grid(
        num_rows=len(WAVES),
        num_cols=1,
        axis_shape=(3.5, 6),
        share_x=True,
        share_y=False,
    )
    for row_index, wave in enumerate(WAVES):
        ax = axs[row_index, 0]
        series = load_wave_convergence(wave=wave)
        plot_wave_panel(ax=ax, series=series)
        add_reference_slope(ax=ax, wave=wave)
        ax.set_ylim(wave["y_lim"][1], wave["y_lim"][0])
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.05,
            y_pos=0.05,
            x_alignment="left",
            y_alignment="bottom",
            label=wave["label"],
        )
        ax.label_outer()  ## with shared x, show tick labels only on the bottom panel
    ## set the shared x-range now, with autoscale off, so no later `ax.plot()` call in the
    ## shared-x group (matplotlib recomputes shared limits from all siblings on each new plot)
    ## can silently revert it: coarser resolution (larger dx) to the left, finer to the right.
    ## Must happen before `add_delta_x_axis` below, which reads this axis's final xlim.
    x_min, x_max = axs[0, 0].get_xlim()
    for ax in axs.flat:
        ax.set_autoscalex_on(False)
        ax.set_xlim(x_max, x_min)
    ncells, cell_sizes = read_resolution_ladder(wave=WAVES[0])
    set_resolution_ticks(ax=axs[-1, 0], ncells=ncells, cell_sizes=cell_sizes)
    add_delta_x_axis(ax=axs[0, 0])
    add_compute_legend(ax=axs[0, 0])
    add_averaging_legend(ax=axs[1, 0])
    add_interpolation_legend(ax=axs[2, 0])
    fig.supylabel(FIELD_LABEL_Y, x=-0.02)
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
