#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $(basename "$0") --sim-dir <path> [--slice] [--profile]" >&2
    exit 1
}

##
## === RESOLVE USER ARGUMENTS
##

SIM_DIR=""
PLOT_SLICE=0
PLOT_PROFILE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim-dir)  SIM_DIR="$2"; shift 2 ;;
        --slice)    PLOT_SLICE=1; shift ;;
        --profile)  PLOT_PROFILE=1; shift ;;
        *)          usage ;;
    esac
done

[[ -n "${SIM_DIR}" ]] || usage

##
## === VALIDATE USER ARGUMENTS
##

if [[ ! -d "${SIM_DIR}" ]]; then
    echo "error: sim dir not found: ${SIM_DIR}" >&2
    exit 1
fi

SIM_DIR="$(cd "${SIM_DIR}" && pwd)"  # resolve to absolute path before cd

if [[ ! -d "${SIM_DIR}/snapshots" ]]; then
    echo "error: snapshots/ not found in ${SIM_DIR}" >&2
    exit 1
fi

##
## === PLOT SIMULATION
##

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # tools/ is one level below repo root

SNAPSHOTS_DIR="${SIM_DIR}/snapshots"
PLOTS_DIR="${SIM_DIR}/plots"
SCALAR_FIELDS=(E_kin E_mag rho pressure)
VECTOR_FIELDS=(mag vel)

uv run --project "${REPO_ROOT}" quokka-plot-vi-evolution \
    --input-dir "${SNAPSHOTS_DIR}" \
    --out-dir "${PLOTS_DIR}/vi-evolution" \
    --fields "${SCALAR_FIELDS[@]}"

if [[ "${PLOT_SLICE}" -eq 1 ]]; then
    uv run --project "${REPO_ROOT}" quokka-plot-slice \
        --input-dir "${SNAPSHOTS_DIR}" \
        --out-dir "${PLOTS_DIR}/slices" \
        --fields "${SCALAR_FIELDS[@]}" "${VECTOR_FIELDS[@]}"
fi

if [[ "${PLOT_PROFILE}" -eq 1 ]]; then
    uv run --project "${REPO_ROOT}" quokka-plot-profile \
        --input-dir "${SNAPSHOTS_DIR}" \
        --out-dir "${PLOTS_DIR}/profiles" \
        --fields "${SCALAR_FIELDS[@]}" "${VECTOR_FIELDS[@]}"
fi
