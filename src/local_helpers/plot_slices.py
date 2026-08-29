## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

from matplotlib import ticker as mpl_ticker
from numpy import typing as numpy_typing

##
## === FUNCTIONS
##


def upsample_slice(
    *,
    array_2d: numpy_typing.NDArray[numpy.floating],
    target_num_cells: int,
) -> numpy_typing.NDArray[numpy.floating]:
    """Block-replicate a coarser array up to `target_num_cells` (not interpolation)."""
    num_rows, num_cols = array_2d.shape
    if (num_rows == target_num_cells) and (num_cols == target_num_cells):
        return array_2d
    scale_row = target_num_cells // num_rows
    scale_col = target_num_cells // num_cols
    return numpy.kron(array_2d, numpy.ones((scale_row, scale_col)))


def make_domain_tick_formatter(
    *,
    labeled_tick_values: tuple[float, ...],
) -> mpl_ticker.FuncFormatter:
    """
    Label only ticks landing on `labeled_tick_values`; every other major tick is drawn unlabeled.

    Labels are math mode, so their minus signs match the ones Matplotlib formats itself.
    """

    def format_tick(
        tick_value: float,
        _tick_position: int,
    ) -> str:
        is_labeled = any(numpy.isclose(tick_value, labeled_value) for labeled_value in labeled_tick_values)
        return f"${tick_value:.2f}$" if is_labeled else ""

    return mpl_ticker.FuncFormatter(format_tick)


## } MODULE
