## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

from numpy import typing as numpy_typing

## personal
from jormi.ww_arrays import mask_2d_arrays
from jormi.ww_plots import manage_figure, plot_data
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_box_positions

##
## === FUNCTIONS
##


def _mask_sarray_slice(
    *,
    sarray: numpy_typing.NDArray[numpy.floating],
    mask: numpy_typing.NDArray[numpy.bool],
) -> numpy_typing.NDArray[numpy.floating]:
    return numpy.where(mask, sarray, numpy.nan)


def plot_comparison_contours(
    *,
    panel: manage_figure.Panel,
    axis_bounds: plot_data.AxisRanges,
    upper_sarray: numpy_typing.NDArray[numpy.floating],
    lower_sarray: numpy_typing.NDArray[numpy.floating],
    contour_value: float,
    upper_color: str,
    lower_color: str,
) -> None:
    """
    Overlay contours of two averaging schemes; split across the off-diagonal, each with a
    faint 'ghost' of the other scheme overlayed.
    """
    num_rows, num_cols = upper_sarray.shape
    upper_mask = mask_2d_arrays.DiagonalMasks2D.get_mask_above_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    lower_mask = mask_2d_arrays.DiagonalMasks2D.get_mask_below_main_diagonal(
        num_rows=num_rows,
        num_cols=num_cols,
    )
    upper_sarray_main = _mask_sarray_slice(
        sarray=upper_sarray,
        mask=upper_mask,
    )
    lower_sarray_main = _mask_sarray_slice(
        sarray=lower_sarray,
        mask=lower_mask,
    )
    upper_sarray_ghost = _mask_sarray_slice(
        sarray=upper_sarray,
        mask=lower_mask,
    )
    lower_sarray_ghost = _mask_sarray_slice(
        sarray=lower_sarray,
        mask=upper_mask,
    )
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(axis_bounds[0][0], axis_bounds[0][1], num_cols),
        numpy.linspace(axis_bounds[1][0], axis_bounds[1][1], num_rows),
    )
    panel.contour(
        grid_x,
        grid_y,
        upper_sarray_main.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.5,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    panel.contour(
        grid_x,
        grid_y,
        lower_sarray_main.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.5,
        alpha=1.0,
        linestyles="solid",
        zorder=1,
    )
    panel.contour(
        grid_x,
        grid_y,
        upper_sarray_ghost.T,
        levels=[contour_value],
        colors=upper_color,
        linewidths=0.45,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    panel.contour(
        grid_x,
        grid_y,
        lower_sarray_ghost.T,
        levels=[contour_value],
        colors=lower_color,
        linewidths=0.45,
        alpha=0.4,
        linestyles="solid",
        zorder=2,
    )
    panel.plot(
        [axis_bounds[0][0], axis_bounds[0][1]],
        [axis_bounds[1][0], axis_bounds[1][1]],
        color="black",
        linewidth=0.4,
        zorder=3,
    )


def add_label(
    *,
    panel: manage_figure.Panel,
    x_position: float,
    y_position: float,
    x_alignment: box_positions.Positions.PositionLike,
    y_alignment: box_positions.Positions.PositionLike,
    label: str,
) -> None:
    x_anchor = validate_box_positions.as_mpl_ha(x_alignment)
    y_anchor = validate_box_positions.as_mpl_va(y_alignment)
    panel.text(
        x_position,
        y_position,
        label,
        transform=panel.transAxes,
        ha=x_anchor.value,
        va=y_anchor.value,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.0",
        },
    )


## } MODULE
