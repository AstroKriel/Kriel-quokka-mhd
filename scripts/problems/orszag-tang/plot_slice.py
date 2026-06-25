## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.mask_2d_arrays import DiagonalMasks2D
from jormi.ww_io import manage_log
from jormi.ww_plots import add_color, manage_plots, plot_data, style_plots

##
## === CONFIGURATION
##

ROOT_DIR = Path(__file__).parents[3]

FIELD_LABEL = r"$\log_{10} |\nabla \times \vec{b}|$"
FIELD_PALETTE = "cmr.wildfire_r"
FIELD_RANGE = (0.0, 3.0)

REL_DIFF_LABEL = r"$(|\nabla \times \vec{b}| - R_{180^\circ}|\nabla \times \vec{b}|) \,/\, |\nabla \times \vec{b}|$"
ABS_DIFF_LABEL = r"$|\nabla \times \vec{b}| - R_{180^\circ}|\nabla \times \vec{b}|$"
DIFF_PALETTE = "cmr.iceburn"
DIFF_PERCENTILE = 99.0
PDF_NUM_BINS = 300
PDF_LINE_COLOR = "#7aa2f7"

AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an OT current-density slice.")
    parser.add_argument("data_path", type=Path, help="Path to the .npy slice file.")
    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument("--rel-diff", action="store_true", default=False, help="Show rot180 relative residual on lower triangle and save PDF plot.")
    diff_group.add_argument("--abs-diff", action="store_true", default=False, help="Show rot180 absolute difference on lower triangle and save PDF plot.")
    args = parser.parse_args()

    data_path = args.data_path.expanduser().resolve()
    stem = data_path.stem
    if args.rel_diff:
        stem += "-rel_diff"
    elif args.abs_diff:
        stem += "-abs_diff"

    figures_dir = ROOT_DIR / "figures" / data_path.relative_to(ROOT_DIR / "datasets").parent
    figures_dir.mkdir(parents=True, exist_ok=True)
    slice_path = figures_dir / (stem + ".png")
    pdf_path = figures_dir / (stem.replace("-slice=", "-pdf=") + ".png")

    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    field = numpy.load(data_path)
    log_field = compute_array_stats.compute_safe_log10(numpy.abs(field))
    field_config = add_color.SequentialConfig(palette_name=FIELD_PALETTE)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(6, 6),
        theme=style_plots.Theme.DARK,
    )
    ax = axs[0, 0]
    if args.rel_diff or args.abs_diff:
        num_rows, num_cols = field.shape
        lower_mask = DiagonalMasks2D.get_mask_below_main_diagonal(num_rows=num_rows, num_cols=num_cols)
        upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(num_rows=num_rows, num_cols=num_cols)
        field_rot = field[::-1, ::-1]
        if args.rel_diff:
            mean_amp = 0.5 * (numpy.abs(field) + numpy.abs(field_rot))
            diff = (field - field_rot) / numpy.where(mean_amp > 0, mean_amp, numpy.nan)
            diff_label = REL_DIFF_LABEL
        else:
            diff = field - field_rot
            diff_label = ABS_DIFF_LABEL
        flat_diff = diff[numpy.isfinite(diff)].ravel()
        bound = float(numpy.percentile(numpy.abs(flat_diff), DIFF_PERCENTILE))
        print(f"{data_path.parent.parent.name}/{data_path.parent.name}  {stem.split('-', 2)[-1]}  bound={bound:.4e}")
        diff_config = add_color.DivergingConfig(mid_value=0.0, palette_name=DIFF_PALETTE)
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=numpy.where(upper_mask, log_field, numpy.nan),
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=FIELD_RANGE,
            palette_config=field_config,
            add_cbar=False,
        )
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=numpy.where(lower_mask, diff, numpy.nan),
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=(-bound, bound),
            palette_config=diff_config,
            add_cbar=False,
        )
        ax.plot(
            [AXIS_BOUNDS[0][0], AXIS_BOUNDS[0][1]],
            [AXIS_BOUNDS[1][0], AXIS_BOUNDS[1][1]],
            color="white",
            linewidth=0.6,
        )
        add_color.add_colorbar(
            ax=ax,
            palette=add_color.make_palette(config=field_config, value_range=FIELD_RANGE),
            label=FIELD_LABEL,
            cbar_side="top",
        )
        add_color.add_colorbar(
            ax=ax,
            palette=add_color.make_palette(config=diff_config, value_range=(-bound, bound)),
            label=diff_label,
            cbar_side="bottom",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        manage_plots.save_figure(fig=fig, fig_path=slice_path, dpi=400)

        ## PDF plot
        clipped = flat_diff[numpy.abs(flat_diff) <= bound]
        pdf = compute_array_stats.estimate_pdf(values=clipped, num_bins=PDF_NUM_BINS)
        fig_pdf, axs_pdf = manage_plots.create_figure_grid(
            num_rows=1,
            num_cols=1,
            axis_shape=(4, 6),
            theme=style_plots.Theme.DARK,
        )
        ax_pdf = axs_pdf[0, 0]
        log_densities = compute_array_stats.compute_safe_log10(pdf.densities)
        log_floor = float(numpy.nanmin(log_densities[numpy.isfinite(log_densities)]))
        ax_pdf.plot(pdf.bin_centers, log_densities, color=PDF_LINE_COLOR, linewidth=1.2)
        ax_pdf.fill_between(pdf.bin_centers, log_densities, log_floor, alpha=0.2, color=PDF_LINE_COLOR)
        ax_pdf.axvline(0.0, color="white", linewidth=0.6, linestyle="--")
        ax_pdf.set_xlabel(diff_label)
        ax_pdf.set_ylabel(r"$\log_{10}(\text{PDF})$")
        ax_pdf.set_xlim(-bound, bound)
        manage_plots.save_figure(fig=fig_pdf, fig_path=pdf_path, dpi=400)
    else:
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=log_field,
            data_format="xy",
            axis_bounds=AXIS_BOUNDS,
            cbar_bounds=FIELD_RANGE,
            palette_config=field_config,
            cbar_label=FIELD_LABEL,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        manage_plots.save_figure(fig=fig, fig_path=slice_path, dpi=400)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
