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

ABS_DIFF_LABEL = r"$|\nabla \times \vec{b}| - R_{180^\circ}|\nabla \times \vec{b}|$"
DIFF_PALETTE = "cmr.iceburn"
DIFF_PERCENTILE = 99.0

AXIS_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5))

##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an OT current-density slice.")
    parser.add_argument("data_path", type=Path, help="Path to the .npy slice file.")
    parser.add_argument("--abs-diff", action="store_true", default=False, help="Show rot180 absolute difference on lower triangle.")
    args = parser.parse_args()

    data_path = args.data_path.expanduser().resolve()
    stem = data_path.stem
    if args.abs_diff:
        stem += "-abs_diff"

    figures_dir = ROOT_DIR / "figures" / data_path.relative_to(ROOT_DIR / "datasets").parent
    figures_dir.mkdir(parents=True, exist_ok=True)
    slice_path = figures_dir / (stem + ".png")

    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    field = numpy.load(data_path)
    log_field = compute_array_stats.compute_safe_log10(numpy.abs(field))
    field_config = add_color.SequentialConfig(palette_name=FIELD_PALETTE)
    fig, axs = manage_plots.create_figure_grid(
        num_rows=1,
        num_cols=1,
        axis_shape=(6, 6),
        theme=style_plots.Theme.LIGHT,
    )
    ax = axs[0, 0]
    if args.abs_diff:
        num_rows, num_cols = field.shape
        lower_mask = DiagonalMasks2D.get_mask_below_main_diagonal(num_rows=num_rows, num_cols=num_cols)
        upper_mask = DiagonalMasks2D.get_mask_above_main_diagonal(num_rows=num_rows, num_cols=num_cols)
        field_rot = field[::-1, ::-1]
        diff = field - field_rot
        diff_label = ABS_DIFF_LABEL
        flat_diff = diff[numpy.isfinite(diff)].ravel()
        bound = float(numpy.percentile(numpy.abs(flat_diff), DIFF_PERCENTILE))
        ## a perfectly symmetric field gives bound == 0, which degenerates the diverging colourbar;
        ## fall back to a unit range so the (uniform-zero) diff panel still renders.
        cbar_bound = bound if bound > 0.0 else 1.0
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
            cbar_bounds=(-cbar_bound, cbar_bound),
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
            palette=add_color.make_palette(config=diff_config, value_range=(-cbar_bound, cbar_bound)),
            label=diff_label,
            cbar_side="bottom",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        manage_plots.save_figure(fig=fig, fig_path=slice_path, dpi=400)
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
