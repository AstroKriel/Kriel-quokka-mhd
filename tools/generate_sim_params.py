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
class GenerateEntry:
    target_dir: Path
    problem_key: str
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
## === PER-PROBLEM ENTRIES
##


def _alfven_wave_circular_convergence_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "alfven-wave-circular/convergence"
    return [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="AlfvenWaveCircular-Convergence",
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in _WAVE_SCHEME_COMBINATIONS
    ]


def _alfven_wave_linear_convergence_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "alfven-wave-linear/convergence/ideal/angle=0-nx=1-ny=0-nz=0"
    return [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="AlfvenWaveLinear-Convergence",
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


def _alfven_wave_linear_correctness_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "alfven-wave-linear/correctness/resistive"
    etas = ("0.00001", "0.0000316", "0.0001", "0.000316", "0.001", "0.00316", "0.01", "0.0316")
    return [
        GenerateEntry(
            target_dir=base / f"eta={eta}/ncells=256/q26-b25-{reconstruction_order_key}",
            problem_key="AlfvenWaveLinear-Correctness",
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


def _fast_wave_convergence_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "fast-wave/convergence/nx=1-ny=0-nz=0"
    return [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="FastWave-Convergence",
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


def _slow_wave_convergence_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "slow-wave/convergence/nx=1-ny=0-nz=0"
    return [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="SlowWave-Convergence",
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


def _slow_wave_correctness_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "slow-wave/correctness/nx=1-ny=2-nz=3"
    ## `ncells=512`'s resolution fields follow the same pattern as the verified `ncells=128` leaf
    ## (blocking_factor/max_grid_size == num_cells); `max_time_steps` is linearly scaled, not verified
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
    entries: list[GenerateEntry] = []
    for ncells, resolution_kwargs in resolutions.items():
        for reconstruction_order_key in ("ppm", "ppm_ep"):
            entries.append(
                GenerateEntry(
                    target_dir=base / f"ncells={ncells}" / f"q26-b25-{reconstruction_order_key}",
                    problem_key="SlowWave-Correctness",
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
    return entries


def _balsara_vortex_entries() -> list[GenerateEntry]:
    resolutions = {
        128: {
            "domain_lo": (-5.0, -5.0, -0.0390625),
            "domain_hi": (5.0, 5.0, 0.0390625),
            "num_cells": (128, 128, 8),
            "blocking_factor": (128, 128, 8),
            "max_grid_size": (128, 128, 128),
            "max_time_steps": 2_000_000,
        },
        64: {
            "domain_lo": (-5.0, -5.0, -0.078125),
            "domain_hi": (5.0, 5.0, 0.078125),
            "num_cells": (64, 64, 8),
            "blocking_factor": (64, 64, 8),
            "max_grid_size": (64, 64, 64),
            "max_time_steps": 800_000,
        },
    }
    entries: list[GenerateEntry] = []
    for ncells, base_kwargs in resolutions.items():
        base = DATASETS_DIR / f"balsara-vortex/ncells={ncells}"
        for reconstruction_order_key in ("ppm", "ppm_ep"):
            entries.append(
                GenerateEntry(
                    target_dir=base / f"q26-b25-{reconstruction_order_key}",
                    problem_key="MHDBalsaraVortex",
                    kwargs={
                        "compute_scheme_key": "q26",
                        "averaging_scheme_key": "b25",
                        "reconstruction_order_key": reconstruction_order_key,
                        **base_kwargs,
                    },
                ),
            )
    return entries


def _blast_wave_entries() -> list[GenerateEntry]:
    base_1024 = DATASETS_DIR / "blast-wave/ncells=1024"
    base_128 = DATASETS_DIR / "blast-wave/ncells=128"
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
        GenerateEntry(
            target_dir=base_1024 / "q26-b25-ppm",
            problem_key="MHDBlast",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm",
                **common_1024,
            },
        ),
        GenerateEntry(
            target_dir=base_1024 / "q26-b25-ppm_ep",
            problem_key="MHDBlast",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_1024,
            },
        ),
        GenerateEntry(
            target_dir=base_128 / "q26-b25-ppm",
            problem_key="MHDBlast",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm",
                **common_128,
            },
        ),
        GenerateEntry(
            target_dir=base_128 / "q26-b25-ppm_ep",
            problem_key="MHDBlast",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_128,
            },
        ),
    ]


def _brio_wu_shock_tube_entries() -> list[GenerateEntry]:
    base_256 = DATASETS_DIR / "brio-wu-shock-tube/ncells=256"
    base_8192 = DATASETS_DIR / "brio-wu-shock-tube/ncells=8192"
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
    entries: list[GenerateEntry] = [
        GenerateEntry(
            target_dir=base_256 / "hlld" /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="BrioWuShockTube",
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
                **common_256,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in hlld_256_combinations
    ]
    entries.append(
        GenerateEntry(
            target_dir=base_256 / "llf" / "q26-b25-ppm_ep",
            problem_key="BrioWuShockTube",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_256,
            },
        ),
    )
    entries.append(
        GenerateEntry(
            target_dir=base_8192 / "hlld" / "q26-b25-ppm_ep",
            problem_key="BrioWuShockTube",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
                **common_8192,
            },
        ),
    )
    return entries


def _ryu_jones_2a_shock_tube_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "ryu-jones-2a-shock-tube/ncells=512"
    hlld_combinations = (
        ("b25", "b25", "ppm_ep"),
        ("b25", "ld04", "ppm_ep"),
        ("fs17", "b25", "ppm_ep"),
        ("fs17", "ld04", "ppm_ep"),
        ("q26", "b25", "ppm"),
        ("q26", "b25", "ppm_ep"),
        ("q26", "ld04", "ppm_ep"),
    )
    entries: list[GenerateEntry] = [
        GenerateEntry(
            target_dir=base / "hlld" /
            f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="RyuJones2aShockTube",
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": reconstruction_order_key,
            },
        ) for compute_scheme_key, averaging_scheme_key, reconstruction_order_key in hlld_combinations
    ]
    entries.append(
        GenerateEntry(
            target_dir=base / "llf" / "q26-b25-ppm_ep",
            problem_key="RyuJones2aShockTube",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ),
    )
    return entries


def _current_sheet_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "current-sheet/ncells=1024"
    entries = [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-{averaging_scheme_key}-ppm",
            problem_key="CurrentSheet",
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": averaging_scheme_key,
                "reconstruction_order_key": "ppm",
            },
        ) for compute_scheme_key in ("b25", "fs17", "q26") for averaging_scheme_key in ("b25", "ld04")
    ]
    entries += [
        GenerateEntry(
            target_dir=base / f"{compute_scheme_key}-b25-ppm_ep",
            problem_key="CurrentSheet",
            kwargs={
                "compute_scheme_key": compute_scheme_key,
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ) for compute_scheme_key in ("b25", "q26")
    ]
    return entries


def _field_loop_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "field-loop/ncells=96"
    return [
        GenerateEntry(
            target_dir=base / "q26-b25-ppm_ep",
            problem_key="FieldLoop",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": "ppm_ep",
            },
        ),
    ]


def _mhd_quirk_entries() -> list[GenerateEntry]:
    base = DATASETS_DIR / "quirk/ncells=128"
    kwargs: dict[str, object] = {
        "compute_scheme_key": "q26",
        "averaging_scheme_key": "b25",
        "reconstruction_order_key": "ppm_ep"
    }
    return [
        GenerateEntry(
            target_dir=base / "q26-b25-ppm_ep",
            problem_key="MHDQuirk",
            kwargs=kwargs,
        ),
        GenerateEntry(
            target_dir=base / "q26-b25-ppm_ep-no-carbuncle-fix",
            problem_key="MHDQuirk",
            kwargs=kwargs,
        ),
    ]


def _orszag_tang_entries() -> list[GenerateEntry]:
    base_1024 = DATASETS_DIR / "orszag-tang/ncells=1024"
    base_4096 = DATASETS_DIR / "orszag-tang/ncells=4096"
    base_8192 = DATASETS_DIR / "orszag-tang/ncells=8192"
    entries: list[GenerateEntry] = [
        GenerateEntry(
            target_dir=base_1024 / f"{compute_scheme_key}-{averaging_scheme_key}-{reconstruction_order_key}",
            problem_key="OrszagTang",
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
    entries += [
        GenerateEntry(
            target_dir=base_4096 / f"q26-b25-{reconstruction_order_key}",
            problem_key="OrszagTang",
            kwargs={
                "compute_scheme_key": "q26",
                "averaging_scheme_key": "b25",
                "reconstruction_order_key": reconstruction_order_key,
                "num_cells": (4096, 4096, 8),
                "max_time_steps": 80_000,
            },
        ) for reconstruction_order_key in ("ppm", "ppm_ep")
    ]
    entries += [
        GenerateEntry(
            target_dir=base_8192 / f"q26-b25-{reconstruction_order_key}",
            problem_key="OrszagTang",
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
    return entries


ALL_ENTRIES: list[GenerateEntry] = [
    *_alfven_wave_circular_convergence_entries(),
    *_alfven_wave_linear_convergence_entries(),
    *_alfven_wave_linear_correctness_entries(),
    *_fast_wave_convergence_entries(),
    *_slow_wave_convergence_entries(),
    *_slow_wave_correctness_entries(),
    *_balsara_vortex_entries(),
    *_blast_wave_entries(),
    *_brio_wu_shock_tube_entries(),
    *_ryu_jones_2a_shock_tube_entries(),
    *_current_sheet_entries(),
    *_field_loop_entries(),
    *_mhd_quirk_entries(),
    *_orszag_tang_entries(),
]

##
## === PROGRAM MAIN
##


def main() -> None:
    for entry in ALL_ENTRIES:
        builder = sim_types.resolve_sim_params_builder(entry.problem_key)
        sim_params = builder(**entry.kwargs)
        sim_params.write(
            output_path=entry.target_dir / "sim_params.toml",
            overwrite=True,
            verbose=False,
        )
    print(f"generated {len(ALL_ENTRIES)} sim_params.toml files; review with `git diff`.")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

## } SCRIPT
