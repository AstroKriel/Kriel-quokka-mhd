#!/usr/bin/env python3
"""
Phase 3 wave convergence sweep driver.

Materialises 108 sim directories and submits SLURM jobs on lanzelot.
108 = 3 waves x 2 orientations x 18 scheme combos (3 EMF x 2 avg x 3 recon).

Usage:
    python run_wave_convergence_sweep.py           # submit all 108 jobs
    python run_wave_convergence_sweep.py --dry-run # print paths only, no sbatch
"""

import itertools
import subprocess
import sys
from pathlib import Path

PAPER_ROOT = Path("/export/ssd/nkriel/quokka/paper")
BUILD_DIR = Path("/export/ssd/nkriel/builds/quokka-cpu")

WAVES = [
    ("alfven-wave-linear-convergence", "AlfvenWaveLinearConvergence", 0.0),
    ("fast-wave-convergence",          "FastWaveConvergence",          90.0),
    ("slow-wave-convergence",          "SlowWaveConvergence",          45.0),
]

ORIENTATIONS = [
    ("grid-aligned", 1, 0, 0),
    ("oblique",      1, 1, 1),
]

EMF_SCHEMES = [
    ("q26",  "Quokka2026"),
    ("fs18", "FelkerStone2017"),
    ("b25",  "Balsara2025"),
]

AVG_SCHEMES = [
    ("ld04", "LondrilloDelZanna2004"),
    ("b25",  "Balsara2025"),
]

RECON_ORDERS = [2, 3, 5]


def make_toml(angle, mx, my, mz, emf_scheme, avg_scheme, recon):
    return (
        "geometry.prob_lo = [0.0, 0.0, 0.0]\n"
        "geometry.prob_hi = [1.0, 1.0, 1.0]\n"
        "geometry.is_periodic = [1, 1, 1]\n"
        "\n"
        "amr.v = 0\n"
        "amr.max_level = 0\n"
        "\n"
        "do_reflux = 0\n"
        "do_subcycle = 0\n"
        "do_tracers = 1\n"
        "plotfile_interval = -1\n"
        "\n"
        "cfl = 0.3\n"
        "\n"
        "hydro.rk_integrator_order = 2\n"
        f"hydro.reconstruction_order = {recon}\n"
        "hydro.use_dual_energy = 0\n"
        "\n"
        f"mhd.reconstruction_order = {recon}\n"
        f'mhd.emf_compute_scheme = "{emf_scheme}"\n'
        f'mhd.emf_averaging_scheme = "{avg_scheme}"\n'
        "\n"
        f"setup.num_modes_x = {mx}\n"
        f"setup.num_modes_y = {my}\n"
        f"setup.num_modes_z = {mz}\n"
        f"setup.angle_between_k_b0 = {angle}\n"
    )


def make_job_script(sim_dir, exec_path, toml_path, job_name):
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --partition=roundtable\n"
        "#SBATCH --nodelist=lanzelot\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=4G\n"
        "#SBATCH --time=01:00:00\n"
        f"#SBATCH --output={sim_dir}/slurm-%j.out\n"
        f"#SBATCH --error={sim_dir}/slurm-%j.err\n"
        "\n"
        f"cd {sim_dir}\n"
        f"mpirun -np 1 {exec_path} {toml_path} > stdout.log 2> stderr.log\n"
    )


def main():
    dry_run = "--dry-run" in sys.argv

    combos = list(itertools.product(WAVES, ORIENTATIONS, EMF_SCHEMES, AVG_SCHEMES, RECON_ORDERS))
    print(f"{len(combos)} sims total", flush=True)

    submitted = 0
    failed = 0

    for (wave_dir, target, angle), (orient_name, mx, my, mz), (emf_tag, emf_scheme), (avg_tag, avg_scheme), recon in combos:
        variant = f"{emf_tag}-{avg_tag}-recon{recon}"
        sim_dir = PAPER_ROOT / wave_dir / orient_name / variant
        exec_path = BUILD_DIR / "src" / "problems" / target / target
        toml_path = sim_dir / "sim_params.toml"
        job_name = f"{wave_dir}-{orient_name}-{variant}"

        sim_dir.mkdir(parents=True, exist_ok=True)
        toml_path.write_text(make_toml(angle, mx, my, mz, emf_scheme, avg_scheme, recon))

        job_script_path = sim_dir / "run.sh"
        job_script_path.write_text(make_job_script(sim_dir, exec_path, toml_path, job_name))
        job_script_path.chmod(0o755)

        if dry_run:
            print(f"  [dry] {sim_dir.relative_to(PAPER_ROOT)}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_script_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  ok  {sim_dir.relative_to(PAPER_ROOT)} -> {result.stdout.strip()}")
            submitted += 1
        else:
            print(f"  ERR {sim_dir.relative_to(PAPER_ROOT)}: {result.stderr.strip()}", file=sys.stderr)
            failed += 1

    if not dry_run:
        print(f"\n{submitted} submitted, {failed} failed")


if __name__ == "__main__":
    main()
