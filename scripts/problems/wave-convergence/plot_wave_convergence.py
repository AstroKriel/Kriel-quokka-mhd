## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import csv_io, manage_io
from jormi.ww_plots import annotate_axis, manage_plots, style_plots

##
## === CONFIGURATION
##

## one panel per linear MHD wave family; each reads the grid-aligned convergence run
WAVES = (
    {
        "label": "Alfvén",
        "convergence_dir": "alfven-wave-linear/convergence/angle=0-nx=1-ny=0-nz=0",
        "csv_name": "alfven_wave_convergence.csv",
    },
    {
        "label": "fast",
        "convergence_dir": "fast-wave/convergence/nx=1-ny=0-nz=0",
        "csv_name": "fast_wave_convergence.csv",
    },
    {
        "label": "slow",
        "convergence_dir": "slow-wave/convergence/nx=1-ny=0-nz=0",
        "csv_name": "slow_wave_convergence.csv",
    },
)

## scheme tokens, matching the dataset directory names: <reconstruction>-<averaging>-<interpolation>
EMF_RECONSTRUCTIONS = ("fs17", "b25", "q26")  ## encoded by colour
EMF_AVERAGINGS = ("ld04", "b25")  ## encoded by marker
## "pcm" (1st order) is kept here for completeness but excluded from the plot for now
INTERPOLATIONS = ("pcm", "plm", "ppm", "ppm_ep")  ## encoded by linestyle; (omitted: "pcm")

COLOR_BY_RECONSTRUCTION = {
    "q26": "cornflowerblue",
    "b25": "orangered",
    "fs17": "forestgreen",
}
MARKER_BY_AVERAGING = {
    "ld04": "D",
    "b25": "o",
}
LINESTYLE_BY_INTERPOLATION = {
    "pcm": "-.",
    "plm": ":",
    "ppm": "--",
    "ppm_ep": "-",
}

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


def load_wave_convergence(
    *,
    wave: dict[str, str],
) -> dict[tuple[str, str, str], tuple[numpy.ndarray, numpy.ndarray]]:
    """Load (dx, error) for every plotted scheme combo of a single wave family."""
    series: dict[tuple[str, str, str], tuple[numpy.ndarray, numpy.ndarray]] = {}
    for reconstruction in EMF_RECONSTRUCTIONS:
        for averaging in EMF_AVERAGINGS:
            for interpolation in INTERPOLATIONS:
                scheme_dir = DATASET_DIR / wave["convergence_dir"] / f"{reconstruction}-{averaging}-{interpolation}"
                table = csv_io.read_csv_file_into_dict(
                    scheme_dir / wave["csv_name"],
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
                cell_size, error = series[(reconstruction, averaging, interpolation)]
                ax.plot(
                    numpy.log10(cell_size),
                    numpy.log10(error),
                    color=COLOR_BY_RECONSTRUCTION[reconstruction],
                    marker=MARKER_BY_AVERAGING[averaging],
                    markersize=5,
                    markerfacecolor="none",
                    linestyle=LINESTYLE_BY_INTERPOLATION[interpolation],
                    linewidth=1.2,
                )
    ax.invert_xaxis()  ## finer resolution (smaller dx) to the right


def read_resolution_ladder(
    *,
    wave: dict[str, str],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Read the shared (ncells, dx) ladder for a wave from a representative combo's CSV."""
    combo = f"{EMF_RECONSTRUCTIONS[0]}-{EMF_AVERAGINGS[0]}-{INTERPOLATIONS[0]}"
    table = csv_io.read_csv_file_into_dict(
        DATASET_DIR / wave["convergence_dir"] / combo / wave["csv_name"],
        verbose=False,
    )
    return numpy.asarray(table["nx"]), numpy.asarray(table["dx"])


def add_resolution_axis(
    *,
    ax,
    ncells: numpy.ndarray,
    cell_sizes: numpy.ndarray,
) -> None:
    """Annotate a top x-axis with the linear cell count at each log10(dx) tick position."""
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())  ## share the (inverted) log10(dx) range of the bottom axis
    top_ax.set_xticks(numpy.log10(cell_sizes))
    top_ax.set_xticklabels([str(int(n)) for n in ncells])
    top_ax.minorticks_off()
    top_ax.set_xlabel(r"$N_{\mathrm{cells}}$")

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
        axis_shape=(3, 6),
        share_x=True,
        share_y=False,
    )
    for row_index, wave in enumerate(WAVES):
        ax = axs[row_index, 0]
        series = load_wave_convergence(wave=wave)
        plot_wave_panel(ax=ax, series=series)
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.05,
            y_pos=0.95,
            x_alignment="left",
            y_alignment="top",
            label=wave["label"],
        )
        ax.label_outer()  ## with shared x, show tick labels only on the bottom panel
    ncells, cell_sizes = read_resolution_ladder(wave=WAVES[0])
    add_resolution_axis(ax=axs[0, 0], ncells=ncells, cell_sizes=cell_sizes)
    axs[-1, 0].set_xlabel(FIELD_LABEL_X)
    fig.supylabel(FIELD_LABEL_Y)
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
