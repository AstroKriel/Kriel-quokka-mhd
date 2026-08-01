## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## personal
from ww_quokka_sims.sim_io.sim_params import sim_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class SimParamsConfig:
    target_dir: Path
    problem_key: sim_types.ProblemKey
    kwargs: dict[str, object]


##
## === CONSTANTS
##

ROOT_DIR: Path = Path(__file__).parents[1]
DATASETS_DIR: Path = ROOT_DIR / "datasets/problems"

## the 3 compute-scheme x 2 averaging-scheme x 4 reconstruction-order grid shared by every wave-convergence sweep
_WAVE_SCHEME_COMBINATIONS: list[tuple[str, str, str]] = [
    (compute_scheme_key, averaging_scheme_key, reconstruction_order_key)
    for compute_scheme_key in ("b25", "fs17", "q26")
    for averaging_scheme_key in ("b25", "ld04")
    for reconstruction_order_key in ("pcm", "plm", "ppm", "ppm_ep")
]

##
## === PER-PROBLEM CONFIGS
##


def alfven_wave_circular_convergence_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "alfven-wave-circular/convergence"
    return [
        SimParamsConfig(
            target_dir=dataset_dir /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ALFVEN_WAVE_CIRCULAR_CONVERGENCE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in _WAVE_SCHEME_COMBINATIONS
    ]


def alfven_wave_linear_convergence_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "alfven-wave-linear/convergence/ideal/angle=0-nx=1-ny=0-nz=0"
    return [
        SimParamsConfig(
            target_dir=dataset_dir /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ALFVEN_WAVE_LINEAR_CONVERGENCE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 0.0,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in _WAVE_SCHEME_COMBINATIONS
    ]


def alfven_wave_linear_correctness_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "alfven-wave-linear/correctness/resistive"
    etas = ("0.00001", "0.0000316", "0.0001", "0.000316", "0.001", "0.00316", "0.01", "0.0316")
    return [
        SimParamsConfig(
            target_dir=dataset_dir / f"eta={eta}/ncells=256/q26-b25-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ALFVEN_WAVE_LINEAR_CORRECTNESS,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": reconstruction_order_key,
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 0.0,
                "stop_time": 5.0,
                "max_time_steps": 100_000,
                "resistivity": float(eta),
            },
        ) for eta in etas for reconstruction_order_key in ("ppm", "ppm_ep")
    ]


def fast_wave_convergence_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "fast-wave/convergence/nx=1-ny=0-nz=0"
    return [
        SimParamsConfig(
            target_dir=dataset_dir /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.FAST_WAVE_CONVERGENCE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 90.0,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in _WAVE_SCHEME_COMBINATIONS
    ]


def slow_wave_convergence_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "slow-wave/convergence/nx=1-ny=0-nz=0"
    return [
        SimParamsConfig(
            target_dir=dataset_dir /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.SLOW_WAVE_CONVERGENCE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 45.0,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in _WAVE_SCHEME_COMBINATIONS
    ]


def slow_wave_correctness_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "slow-wave/correctness/nx=1-ny=2-nz=3"
    resolutions = {
        128: {
            "num_cells": (128, 128, 128),
            "blocking_factor": 128,
            "max_grid_size": 128,
            "max_time_steps": 100_000,
        },
        512: {
            "num_cells": (512, 512, 512),
            "blocking_factor": 512,
            "max_grid_size": 512,
            "max_time_steps": 400_000,
        },
    }
    configs: list[SimParamsConfig] = []
    for ncells, resolution_kwargs in resolutions.items():
        for reconstruction_order_key in ("ppm", "ppm_ep"):
            configs.append(
                SimParamsConfig(
                    target_dir=dataset_dir / f"ncells={ncells}" / f"q26-b25-{reconstruction_order_key}",
                    problem_key=sim_types.ProblemKey.SLOW_WAVE_CORRECTNESS,
                    kwargs={
                        "compute_scheme_key": "q26",
                        "averaging_scheme_key": "b25",
                        "reconstruction_order_key": reconstruction_order_key,
                        "num_modes_x": 1,
                        "num_modes_y": 2,
                        "num_modes_z": 3,
                        "angle_between_k_b0": 45.0,
                        "stop_time": 0.987669,
                        **resolution_kwargs,
                    },
                ),
            )
    return configs


def balsara_vortex_configs() -> list[SimParamsConfig]:

    def _compute_domain(
        num_cells_xy: int,
    ) -> dict[str, tuple[float, float, float]]:
        """The z-domain spans the width of one cell in the (x,y) plane, and is split across 8 cells."""
        z_extent = 5.0 / num_cells_xy
        return {"domain_lo": (-5.0, -5.0, -z_extent), "domain_hi": (5.0, 5.0, z_extent)}

    resolutions = {
        128: {
            "num_cells": (128, 128, 8),
            "blocking_factor": (128, 128, 8),
            "max_grid_size": (128, 128, 128),
            "max_time_steps": 2_000_000,
        },
        64: {
            "num_cells": (64, 64, 8),
            "blocking_factor": (64, 64, 8),
            "max_grid_size": (64, 64, 64),
            "max_time_steps": 800_000,
        },
    }
    configs: list[SimParamsConfig] = []
    for ncells, base_kwargs in resolutions.items():
        dataset_dir = DATASETS_DIR / f"balsara-vortex/ncells={ncells}"
        for reconstruction_order_key in ("ppm", "ppm_ep"):
            configs.append(
                SimParamsConfig(
                    target_dir=dataset_dir / f"q26-b25-{reconstruction_order_key}",
                    problem_key=sim_types.ProblemKey.MHD_BALSARA_VORTEX,
                    kwargs={
                        "compute_scheme_key": "q26",
                        "averaging_scheme_key": "b25",
                        "reconstruction_order_key": reconstruction_order_key,
                        **_compute_domain(ncells),
                        **base_kwargs,
                    },
                ),
            )
    return configs


def blast_wave_configs() -> list[SimParamsConfig]:
    dataset_dir_1024 = DATASETS_DIR / "blast-wave/ncells=1024"
    dataset_dir_128 = DATASETS_DIR / "blast-wave/ncells=128"
    common_1024 = {
        "num_cells": (1024, 1024, 1024),
        "blocking_factor": (32, 32, 32),
        "max_grid_size": 128,
        "max_time_steps": 10_000,
        "snapshot_time_interval": 0.0025,
        "checkpoint_time_interval": 0.005,
        "checkpoint_prefix": "checkpoints/chk",
    }
    common_128 = {
        "num_cells": (128, 128, 128),
        "blocking_factor": (16, 16, 16),
        "max_grid_size": (16, 16, 16),
        "max_time_steps": 4_000,
        "snapshot_index_interval": 25,
    }
    return [
        SimParamsConfig(
            target_dir=dataset_dir_1024 / "q26-b25-ppm",
            problem_key=sim_types.ProblemKey.MHD_BLAST,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm",
                **common_1024,
            },
        ),
        SimParamsConfig(
            target_dir=dataset_dir_1024 / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.MHD_BLAST,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_1024,
            },
        ),
        SimParamsConfig(
            target_dir=dataset_dir_128 / "q26-b25-ppm",
            problem_key=sim_types.ProblemKey.MHD_BLAST,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm",
                **common_128,
            },
        ),
        SimParamsConfig(
            target_dir=dataset_dir_128 / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.MHD_BLAST,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_128,
            },
        ),
    ]


def brio_wu_shock_tube_configs() -> list[SimParamsConfig]:
    dataset_dir_256 = DATASETS_DIR / "brio-wu-shock-tube/ncells=256"
    dataset_dir_8192 = DATASETS_DIR / "brio-wu-shock-tube/ncells=8192"
    common_256 = {
        "num_cells": (256, 8, 8),
        "blocking_factor": (16, 8, 8),
        "max_grid_size": 128,
        "max_time_steps": 4_000,
        "snapshot_index_interval": 200,
    }
    common_8192 = {
        "num_cells": (8192, 8, 8),
        "blocking_factor": (16, 8, 8),
        "max_grid_size": 64,
        "max_time_steps": 100_000,
        "snapshot_index_interval": 1600,
    }
    hlld_256_combinations = (
        ("b25", "b25", "ppm_ep"),
        ("b25", "ld04", "ppm_ep"),
        ("fs17", "b25", "ppm_ep"),
        ("fs17", "ld04", "ppm_ep"),
        ("q26", "b25", "ppm"),
        ("q26", "b25", "ppm_ep"),
        ("q26", "ld04", "ppm_ep"),
    )
    configs: list[SimParamsConfig] = [
        SimParamsConfig(
            target_dir=dataset_dir_256 / "hlld" /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.BRIO_WU_SHOCK_TUBE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
                **common_256,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in hlld_256_combinations
    ]
    configs.append(
        SimParamsConfig(
            target_dir=dataset_dir_256 / "llf" / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.BRIO_WU_SHOCK_TUBE,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_256,
            },
        ),
    )
    configs.append(
        SimParamsConfig(
            target_dir=dataset_dir_8192 / "hlld" / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.BRIO_WU_SHOCK_TUBE,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_8192,
            },
        ),
    )
    return configs


def ryu_jones_2a_shock_tube_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "ryu-jones-2a-shock-tube/ncells=512"
    hlld_combinations = (
        ("b25", "b25", "ppm_ep"),
        ("b25", "ld04", "ppm_ep"),
        ("fs17", "b25", "ppm_ep"),
        ("fs17", "ld04", "ppm_ep"),
        ("q26", "b25", "ppm"),
        ("q26", "b25", "ppm_ep"),
        ("q26", "ld04", "ppm_ep"),
    )
    configs: list[SimParamsConfig] = [
        SimParamsConfig(
            target_dir=dataset_dir / "hlld" /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.RYU_JONES_2A_SHOCK_TUBE,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in hlld_combinations
    ]
    configs.append(
        SimParamsConfig(
            target_dir=dataset_dir / "llf" / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.RYU_JONES_2A_SHOCK_TUBE,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ),
    )
    return configs


def current_sheet_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "current-sheet/ncells=1024"
    configs = [
        SimParamsConfig(
            target_dir=dataset_dir / f"{compute_scheme_key}-{averaging_scheme_key}-ppm",
            problem_key=sim_types.ProblemKey.CURRENT_SHEET,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": "ppm",
            },
        ) for compute_scheme_key in ("b25", "fs17", "q26") for averaging_scheme_key in ("b25", "ld04")
    ]
    configs += [
        SimParamsConfig(
            target_dir=dataset_dir / f"{compute_scheme_key}-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.CURRENT_SHEET,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ) for compute_scheme_key in ("b25", "q26")
    ]
    return configs


def field_loop_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "field-loop/ncells=96"
    return [
        SimParamsConfig(
            target_dir=dataset_dir / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.FIELD_LOOP,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ),
    ]


def mhd_quirk_configs() -> list[SimParamsConfig]:
    dataset_dir = DATASETS_DIR / "quirk/ncells=128"
    kwargs: dict[str, object] = {
        "compute_scheme_key": "q26",
        "averaging_scheme_key": "b25",
        "reconstruction_order_key": "ppm_ep",
    }
    return [
        SimParamsConfig(
            target_dir=dataset_dir / "q26-b25-ppm_ep",
            problem_key=sim_types.ProblemKey.MHD_QUIRK,
            kwargs=kwargs,
        ),
        SimParamsConfig(
            target_dir=dataset_dir / "q26-b25-ppm_ep-no-carbuncle-fix",
            problem_key=sim_types.ProblemKey.MHD_QUIRK,
            kwargs=kwargs,
        ),
    ]


def orszag_tang_configs() -> list[SimParamsConfig]:
    dataset_dir_1024 = DATASETS_DIR / "orszag-tang/ncells=1024"
    dataset_dir_4096 = DATASETS_DIR / "orszag-tang/ncells=4096"
    dataset_dir_8192 = DATASETS_DIR / "orszag-tang/ncells=8192"
    configs: list[SimParamsConfig] = [
        SimParamsConfig(
            target_dir=dataset_dir_1024 /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ORSZAG_TANG,
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
            },
        )
        for compute_scheme_key in ("b25", "fs17", "q26")
        for averaging_scheme_key in ("b25", "ld04")
        for reconstruction_order_key in ("plm", "ppm", "ppm_ep")
    ]
    configs += [
        SimParamsConfig(
            target_dir=dataset_dir_4096 / f"q26-b25-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ORSZAG_TANG,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": reconstruction_order_key,
                "num_cells": (4096, 4096, 8),
                "max_time_steps": 80_000,
            },
        ) for reconstruction_order_key in ("ppm", "ppm_ep")
    ]
    configs += [
        SimParamsConfig(
            target_dir=dataset_dir_8192 / f"q26-b25-{reconstruction_order_key}",
            problem_key=sim_types.ProblemKey.ORSZAG_TANG,
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": reconstruction_order_key,
                "num_cells": (8192, 8192, 8),
                "max_time_steps": 200_000,
                "use_reflux": 0,
                "use_subcycle": 0,
                "checkpoint_time_interval": 0.05,
                "checkpoint_prefix": "checkpoints/chk",
            },
        ) for reconstruction_order_key in ("ppm", "ppm_ep")
    ]
    return configs


ALL_CONFIGS: list[SimParamsConfig] = [
    *alfven_wave_circular_convergence_configs(),
    *alfven_wave_linear_convergence_configs(),
    *alfven_wave_linear_correctness_configs(),
    *fast_wave_convergence_configs(),
    *slow_wave_convergence_configs(),
    *slow_wave_correctness_configs(),
    *balsara_vortex_configs(),
    *blast_wave_configs(),
    *brio_wu_shock_tube_configs(),
    *ryu_jones_2a_shock_tube_configs(),
    *current_sheet_configs(),
    *field_loop_configs(),
    *mhd_quirk_configs(),
    *orszag_tang_configs(),
]

##
## === PROGRAM MAIN
##


def main() -> None:
    for config in ALL_CONFIGS:
        builder = sim_types.resolve_sim_params_builder(config.problem_key)
        sim_params = builder(**config.kwargs)
        sim_params.write(
            output_path=config.target_dir / "sim_params.toml",
            overwrite=True,
            verbose=False,
        )
    print(f"generated {len(ALL_CONFIGS)} sim_params.toml files; review with `git diff`.")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
