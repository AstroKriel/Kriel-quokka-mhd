## { MODULE

##
## === DEPENDENCIES
##

import numpy
import argparse

from pathlib import Path
from matplotlib.figure import Figure as mpl_Figure

from jormi.utils import list_utils
from jormi.ww_types import type_checks
from jormi.ww_plots import plot_manager
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_type

##
## === QUOKKA FIELDS
##

QUOKKA_FIELD_LOOKUP = {
    "rho": {
        "loader": "load_3d_density_sfield",
        "cmap": "Greys",
    },
    "vel": {
        "loader": "load_3d_velocity_vfield",
        "cmap": "Blues",
    },
    "vel_magn": {
        "loader": "load_3d_velocity_magnitude_sfield",
        "cmap": "Blues",
    },
    "mag": {
        "loader": "load_3d_magnetic_vfield",
        "cmap": "Oranges",
    },
    "Etot": {
        "loader": "load_3d_total_energy_sfield",
        "cmap": "cividis",
    },
    "Ekin": {
        "loader": "load_3d_kinetic_energy_sfield",
        "cmap": "magma",
    },
    "Ekin_div": {
        "loader": "load_3d_div_kinetic_energy_sfield",
        "cmap": "magma",
    },
    "Ekin_sol": {
        "loader": "load_3d_sol_kinetic_energy_sfield",
        "cmap": "magma",
    },
    "Ekin_bulk": {
        "loader": "load_3d_bulk_kinetic_energy_sfield",
        "cmap": "magma",
    },
    "Emag": {
        "loader": "load_3d_magnetic_energy_sfield",
        "cmap": "plasma",
    },
    "Eint": {
        "loader": "load_3d_internal_energy_sfield",
        "cmap": "magma",
    },
    "pressure": {
        "loader": "load_3d_pressure_sfield",
        "cmap": "Purples",
    },
    "divb": {
        "loader": "load_3d_divb_sfield",
        "cmap": "bwr",
    },
    "cur": {
        "loader": "load_current_density_sfield",
        "cmap": "cubehelix",
    },
}

##
## === HELPER FUNCTIONS
##


def get_sim_time(
    field: field_type.ScalarField_3D | field_type.VectorField_3D,
) -> float:
    sim_time = field.sim_time
    type_checks.ensure_finite_float(
        param=sim_time,
        param_name="sim_time",
        allow_none=False,
    )
    assert sim_time is not None
    return float(sim_time)


def as_latex_label(
    label: str,
) -> str:
    if "$" in label:
        return label
    return f"${label}$"


def validate_fields(
    fields_to_plot: list[str] | tuple[str, ...] | None,
) -> None:
    valid_fields = set(QUOKKA_FIELD_LOOKUP.keys())
    if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
        raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")


def base_parser() -> argparse.ArgumentParser:
    """
    Shared parser arguments for diagnostic scripts.
    
    Use as a parent:
        parser = argparse.ArgumentParser(parents=[utils.base_parser()], description="...")
    """
    field_list = list_utils.as_string(elems=sorted(QUOKKA_FIELD_LOOKUP.keys()))
    axis_list = list_utils.as_string(elems=list(cartesian_axes.VALID_3D_AXIS_LABELS))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Path to a Quokka simulation or dataset directory.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        default="plt",
        help="Dataset tag (e.g. `plt` -> plt00010, plt00020). Default: `plt`.",
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"Fields to plot. Options: {field_list}",
    )
    parser.add_argument(
        "--comps",
        "-c",
        nargs="+",
        default=None,
        help=f"Vector field components to show. Options: {axis_list}",
    )
    parser.add_argument(
        "--axes",
        "-a",
        nargs="+",
        default=None,
        help=f"Axes to slice along. Options: {axis_list}",
    )
    return parser


def create_figure(
    num_rows: int = 1,
    num_cols: int = 1,
    add_cbar_space: bool = False,
) -> tuple[mpl_Figure, plot_manager.PlotAxesArray]:
    if (num_rows == 1) and (num_cols == 1):
        fig, ax = plot_manager.create_figure()
        if add_cbar_space:
            fig.subplots_adjust(right=0.82)
        axs_grid = numpy.asarray([[ax]], dtype=object)
        return fig, axs_grid
    fig, axs_grid = plot_manager.create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        y_spacing=0.25,
        x_spacing=0.75 if add_cbar_space else 0.25,
    )
    return fig, axs_grid


def looks_like_boxlib_dir(
    dataset_dir: Path,
) -> bool:
    type_checks.ensure_type(param=dataset_dir, valid_types=Path)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False
    has_header = (dataset_dir / "Header").is_file()
    has_level0 = (dataset_dir / "Level_0").is_dir()
    return has_header and has_level0


def get_latest_dataset_dirs(
    sim_dir: Path,
    dataset_tag: str,
) -> list[Path]:
    dataset_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and (dataset_tag in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    dataset_dirs.sort(
        key=lambda dataset_dir: int(get_dataset_index_string(dataset_dir, dataset_tag)),
    )
    return dataset_dirs


def resolve_dataset_dirs(
    input_dir: Path,
    dataset_tag: str,
    max_elems: int | None = None,
) -> list[Path]:
    if (dataset_tag in input_dir.name) or looks_like_boxlib_dir(input_dir):
        return [input_dir]
    dataset_dirs = get_latest_dataset_dirs(
        sim_dir=input_dir,
        dataset_tag=dataset_tag,
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found using tag `{dataset_tag}` under: {input_dir}")
    if max_elems is not None:
        dataset_dirs = list_utils.sample_list(
            elems=dataset_dirs,
            max_elems=max_elems,
        )
    return dataset_dirs


def get_dataset_index_string(
    dataset_dir: Path,
    dataset_tag: str,
) -> str:
    dataset_name = dataset_dir.name
    if dataset_tag not in dataset_name:
        raise ValueError(f"Dataset tag `{dataset_tag}` was not found in `{dataset_name}`.")
    name_parts = dataset_name.split(dataset_tag)
    if len(name_parts) < 2:
        raise ValueError(f"Unexpected dataset name format: {dataset_name}")
    digits_string = name_parts[1].split(".")[0]
    if not digits_string.isdigit():
        raise ValueError(f"Expected digits after `{dataset_tag}` in {dataset_name}")
    return digits_string


def get_max_index_width(
    dataset_dirs: list[Path],
    dataset_tag: str,
) -> int:
    if not dataset_dirs: return 1
    index_widths: list[int] = []
    for dataset_dir in dataset_dirs:
        dataset_index_string = get_dataset_index_string(
            dataset_dir=dataset_dir,
            dataset_tag=dataset_tag,
        )
        index_widths.append(len(dataset_index_string))
    return max(index_widths) if len(index_widths) > 0 else 1


## } MODULE
