## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import pandas

## personal
from jormi.ww_io import csv_io
from jormi.ww_io import manage_log

##
## === CONSTANTS
##

DATASETS_DIR: Path = Path(__file__).parents[2] / "datasets"
ELIZABETH_DIR: Path = DATASETS_DIR / "elizabeth.gitignored"

COMPUTE_SCHEME_MAP: dict[str, str] = {
    "Quokka2026": "Q26",
    "FelkerStone2017": "FS18",
    "Balsara2025": "B25",
}
AVERAGING_SCHEME_MAP: dict[str, str] = {
    "LondrilloDelZanna2004": "LD04",
    "Balsara2025": "B25",
}

##
## === EXTRACT SCALING DATA
##

def extract_strong_scaling() -> None:
    df = pandas.read_csv(ELIZABETH_DIR / "strong_scaling_all.csv")
    df = df[["num_gpus", "us_per_zone_update", "compute_scheme", "averaging_scheme"]].copy()
    df["compute_scheme"] = df["compute_scheme"].map(COMPUTE_SCHEME_MAP)
    df["averaging_scheme"] = df["averaging_scheme"].map(AVERAGING_SCHEME_MAP)
    df = df.sort_values(["compute_scheme", "averaging_scheme", "num_gpus"]).reset_index(drop=True)
    csv_io.save_dict_to_csv_file(
        file_path=DATASETS_DIR / "scalings" / "strong-scaling.csv",
        input_dict=df.to_dict(orient="list"),
        overwrite=True,
    )


def extract_weak_scaling() -> None:
    df = pandas.read_csv(ELIZABETH_DIR / "weak_scaling_all.csv")
    df = df[["num_gpus", "us_per_zone_update", "compute_scheme", "averaging_scheme"]].copy()
    df["compute_scheme"] = df["compute_scheme"].map(COMPUTE_SCHEME_MAP)
    df["averaging_scheme"] = df["averaging_scheme"].map(AVERAGING_SCHEME_MAP)
    df = df.sort_values(["compute_scheme", "averaging_scheme", "num_gpus"]).reset_index(drop=True)
    csv_io.save_dict_to_csv_file(
        file_path=DATASETS_DIR / "scalings" / "weak-scaling.csv",
        input_dict=df.to_dict(orient="list"),
        overwrite=True,
    )


##
## === PROGRAM MAIN
##

def main() -> None:
    manage_log.log_section(title="strong scaling")
    extract_strong_scaling()
    manage_log.log_section(title="weak scaling")
    extract_weak_scaling()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
