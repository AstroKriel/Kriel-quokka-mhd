#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $(basename "$0") --sim-dir <path> --exe-path <path> --num-procs <n>" >&2
    exit 1
}

##
## === RESOLVE USER ARGUMENTS
##

SIM_DIR=""
EXE_PATH=""
NUM_PROCS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim-dir)   SIM_DIR="$2";   shift 2 ;;
        --exe-path)  EXE_PATH="$2";  shift 2 ;;
        --num-procs) NUM_PROCS="$2"; shift 2 ;;
        *)           usage ;;
    esac
done

[[ -n "${SIM_DIR}" && -n "${EXE_PATH}" && -n "${NUM_PROCS}" ]] || usage

##
## === VALIDATE USER ARGUMENTS
##

if [[ ! -d "${SIM_DIR}" ]]; then
    echo "error: sim dir not found: ${SIM_DIR}" >&2
    exit 1
fi

SIM_DIR="$(cd "${SIM_DIR}" && pwd)"  # resolve to absolute path before cd
PARAMS_FILE="${SIM_DIR}/sim_params.toml"

if [[ ! -f "${PARAMS_FILE}" ]]; then
    echo "error: sim_params.toml not found in ${SIM_DIR}" >&2
    exit 1
fi

if [[ ! -x "${EXE_PATH}" ]]; then
    echo "error: exe not found or not executable: ${EXE_PATH}" >&2
    exit 1
fi

##
## === RUN SIMULATION
##

cd "${SIM_DIR}"
mkdir -p snapshots

if compgen -G "snapshots/plt_*" > /dev/null 2>&1; then
    echo "error: snapshots/ directory already contains data; delete snapshots/plt_* before re-running." >&2
    exit 1
fi

SEP="$(printf '-%.0s' {1..50})"  # 50 chars of "-"

{
    echo "${SEP}"
    echo "exe:        ${EXE_PATH}"
    echo "params:     ${PARAMS_FILE}"
    echo "num_procs:  ${NUM_PROCS}"
    echo "started:    $(date)"
    echo "${SEP}"
    time mpirun -n "${NUM_PROCS}" "${EXE_PATH}" "${PARAMS_FILE}" || true  # flush log before exit (even if it crashes)
    echo "${SEP}"
    echo "finished:   $(date)"
    echo "${SEP}"
} 2>&1 | tee snapshots/sim.log

## .
