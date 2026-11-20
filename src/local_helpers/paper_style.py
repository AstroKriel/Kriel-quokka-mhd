## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_io import manage_log
from jormi.ww_plots import style_figure

##
## === FIGURE PARAMS
##

_default_text_sizes = style_figure.TextSizeParams()

TEXT_SIZE_PARAMS = style_figure.TextSizeParams(
    largest_size_pt=13.0,
    legend_level=_default_text_sizes.annotation_level,
)

FIGURE_PARAMS = style_figure.FigureParams(
    text_size_params=TEXT_SIZE_PARAMS,
)

##
## === PAGE LAYOUT
##

## widths the compiled paper reports for `\the\textwidth` and `\the\columnwidth`, in TeX
## points (1/72.27 inch, unlike the 1/72 inch pt that Matplotlib and jormi use)
_TEX_PT_PER_INCH = 72.27
_TEXT_WIDTH_TEX_PT = 513.12
_COLUMN_WIDTH_TEX_PT = 245.27
TEXT_WIDTH_CM = _TEXT_WIDTH_TEX_PT / _TEX_PT_PER_INCH * style_figure.CM_PER_INCH
COLUMN_WIDTH_CM = _COLUMN_WIDTH_TEX_PT / _TEX_PT_PER_INCH * style_figure.CM_PER_INCH


def page_width(
    *,
    fraction: float,
) -> style_figure.FigureWidth:
    """Width of a `figure*` placed at `fraction` of `\\linewidth`, so it is drawn at the size it prints."""
    return style_figure.FigureWidth(
        max_width_cm=TEXT_WIDTH_CM,
        width_fraction=fraction,
    )


def column_width(
    *,
    fraction: float,
) -> style_figure.FigureWidth:
    """Width of a `figure` placed at `fraction` of `\\linewidth`, so it is drawn at the size it prints."""
    return style_figure.FigureWidth(
        max_width_cm=TEXT_WIDTH_CM,
        width_fraction=fraction * COLUMN_WIDTH_CM / TEXT_WIDTH_CM,
    )


##
## === SETUP
##


def setup_plotting_script(
    *,
    figure_params: style_figure.FigureParams = FIGURE_PARAMS,
) -> None:
    """
    Apply the paper's log formatting and figure style; call once at the start of a script's `main`.

    A script that fine-tunes the style passes `dataclasses.replace(FIGURE_PARAMS, ...)`.
    """
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params(figure_params=figure_params)


## } MODULE
