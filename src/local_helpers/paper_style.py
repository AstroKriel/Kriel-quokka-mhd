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

FIGURE_PARAMS = style_figure.FigureParams(
    text_size_params=style_figure.TextSizeParams(
        legend_level=_default_text_sizes.annotation_level,
    ),
)

##
## === SETUP
##


def setup_plotting_script() -> None:
    """Apply the paper's log formatting and figure style; call once at the start of a script's `main`."""
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params(figure_params=FIGURE_PARAMS)


## } MODULE
