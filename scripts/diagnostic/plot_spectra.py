## { SCRIPT

##
## === DEPENDENCIES
##

import argparse
import numpy

from pathlib import Path
from dataclasses import dataclass

from jormi.ww_plots import plot_manager, add_color
from jormi.ww_fields.fields_3d import field_type, compute_spectra
from jormi.ww_types import type_checks, array_checks

from ww_quokka_sims.sim_io import load_dataset

import utils

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class SpectraData:
    sim_time: float
    field_label: str
    k_bin_centers: numpy.ndarray
    log10_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_array(array=self.k_bin_centers)
        array_checks.ensure_array(array=self.log10_spectrum)
        array_checks.ensure_1d(array=self.k_bin_centers)
        array_checks.ensure_1d(array=self.log10_spectrum)
        array_checks.ensure_same_shape(
            array_a=self.k_bin_centers,
            array_b=self.log10_spectrum,
        )


##
## === OPERATOR CLASSES
##


class ComputeSpectra:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: str,
    ):
        self.dataset_dirs = dataset_dirs
        self.field_name = field_name
        self.field_loader = field_loader

    @staticmethod
    def _get_sim_time(
        *,
        field: field_type.ScalarField_3D,
    ) -> float:
        sim_time = field.sim_time
        type_checks.ensure_finite_float(
            param=sim_time,
            param_name="sim_time",
            allow_none=False,
        )
        assert sim_time is not None
        return float(sim_time)

    def run(
        self,
    ) -> list[SpectraData]:
        field_spectra: list[SpectraData] = []
        for dataset_dir in self.dataset_dirs:
            with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
                loader_fn = getattr(ds, self.field_loader)
                field = loader_fn()
            if not isinstance(field, field_type.ScalarField_3D):
                raise TypeError(
                    f"`{self.field_name}` is not a scalar field; "
                    "power spectra are only supported for scalar fields.",
                )
            sim_time = self._get_sim_time(field=field)
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(field)
            log10_spectrum = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.spectrum_1d,
                    value=0.0,
                ),
            )
            field_spectra.append(
                SpectraData(
                    sim_time=sim_time,
                    field_label=field_type.get_label(field),
                    k_bin_centers=spectrum.k_bin_centers_1d,
                    log10_spectrum=log10_spectrum,
                ),
            )
        field_spectra.sort(key=lambda s: s.sim_time)
        return field_spectra


class RenderSpectra:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        fig_dir: Path,
        field_name: str,
        field_loader: str,
        cmap_name: str,
    ):
        self.dataset_dirs = dataset_dirs
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.field_loader = field_loader
        self.cmap_name = cmap_name

    @staticmethod
    def _style_ax(
        *,
        ax,
        field_label: str,
    ) -> None:
        ax.set_xlabel(r"$k$")
        raw_field_label = field_label.replace("$", "")
        ax.set_ylabel(rf"$\log_{{10}}\big(\mathcal{{P}}_{{{raw_field_label}}}(k)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        ax,
        spectra_data: SpectraData,
        color,
    ) -> None:
        ax.plot(
            spectra_data.k_bin_centers,
            spectra_data.log10_spectrum,
            lw=2.0,
            color=color,
        )

    @staticmethod
    def _plot_series(
        *,
        axs_grid,
        field_spectra: list[SpectraData],
        cmap_name: str,
    ) -> None:
        ax = axs_grid[0][0]
        cmap, norm = add_color.create_cmap(
            cmap_name=cmap_name,
            min_cmap_value=0.25,
            vmin=0,
            vmax=max(
                0,
                len(field_spectra) - 1,
            ),
        )
        for series_index, spectra_data in enumerate(field_spectra):
            color = cmap(norm(series_index))
            RenderSpectra._plot_snapshot(
                ax=ax,
                spectra_data=spectra_data,
                color=color,
            )
        add_color.add_cbar_from_cmap(
            ax=axs_grid[-1][-1],
            label=r"snapshot index",
            cmap=cmap,
            norm=norm,
            anchor_side="right",
            ax_percentage=0.05,
        )

    def run(
        self,
    ) -> None:
        compute = ComputeSpectra(
            dataset_dirs=self.dataset_dirs,
            field_name=self.field_name,
            field_loader=self.field_loader,
        )
        field_spectra = compute.run()
        if not field_spectra:
            return
        fig, axs_grid = utils.create_figure(
            num_rows=1,
            num_cols=1,
            add_cbar_space=len(field_spectra) > 1,
        )
        ax = axs_grid[0][0]
        if len(field_spectra) == 1:
            self._plot_snapshot(
                ax=ax,
                spectra_data=field_spectra[0],
                color="black",
            )
        else:
            self._plot_series(
                axs_grid=axs_grid,
                field_spectra=field_spectra,
                cmap_name=self.cmap_name,
            )
        self._style_ax(
            ax=ax,
            field_label=field_spectra[0].field_label,
        )
        fig_path = self.fig_dir / f"{self.field_name}_spectra.png"
        plot_manager.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset_tag: str,
        fields_to_plot: tuple[str, ...] | list[str] | None,
    ):
        type_checks.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        valid_fields = set(utils.QUOKKA_FIELD_LOOKUP.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")
        self.input_dir = Path(input_dir)
        self.dataset_tag = dataset_tag
        self.fields_to_plot = type_checks.as_tuple(param=fields_to_plot)

    def run(
        self,
    ) -> None:
        dataset_dirs = utils.resolve_dataset_dirs(
            input_dir=self.input_dir,
            dataset_tag=self.dataset_tag,
        )
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = utils.QUOKKA_FIELD_LOOKUP[field_name]
            renderer = RenderSpectra(
                dataset_dirs=dataset_dirs,
                fig_dir=fig_dir,
                field_name=field_name,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
            )
            renderer.run()


##
## === PROGRAM MAIN
##


def main():
    user_args = argparse.ArgumentParser(
        description="Plot power spectra of Quokka scalar fields.",
        parents=[utils.base_parser()],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
