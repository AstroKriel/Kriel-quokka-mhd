## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import subprocess
import sys
import time

from pathlib import Path

## local
from jormi.ww_io import manage_log

##
## === MAIN PROGRAM
##


def main():
    ## discover all validation scripts
    validation_root = Path(__file__).parent
    test_scripts = sorted(
        validation_root.rglob(
            "test_*.py",
        ),
    )
    if not test_scripts:
        manage_log.log_alert(text="No validation scripts found.")
        sys.exit(0)
    manage_log.log_section(
        title="Validation Suite",
        show_time=True,
    )
    manage_log.log_empty_lines()
    ## run each script as a subprocess and collect results
    results: list[tuple[str, bool, float]] = []
    for script_path in test_scripts:
        test_label = str(
            script_path.relative_to(
                validation_root,
            ),
        )
        manage_log.log_task(
            text=test_label,
            show_time=False,
        )
        start_time = time.perf_counter()
        test = subprocess.run(
            args=[sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        elapsed_time = time.perf_counter() - start_time
        test_passed = test.returncode == 0
        results.append((test_label, test_passed, elapsed_time))
        if test_passed:
            manage_log.log_action(
                title=test_label,
                outcome=manage_log.ActionOutcome.SUCCESS,
                notes={"elapsed": f"{elapsed_time:.2f}s"},
            )
        else:
            test_output = (test.stdout + test.stderr).strip()
            manage_log.log_action(
                title=test_label,
                outcome=manage_log.ActionOutcome.FAILURE,
                notes={
                    "elapsed": f"{elapsed_time:.2f}s",
                    "test_output": test_output[:500] if test_output else "(no test_output)",
                },
            )
    ## print summary and exit with non-zero code if any script failed
    num_tests_passed = sum(1 for (_, _test_passed, _) in results if _test_passed)
    total_tests = len(results)
    total_elapsed_time = sum(_elapsed_time for (_, _, _elapsed_time) in results)
    manage_log.log_summary(
        title="Validation Results",
        notes={
            _test_label: f"{'pass' if _test_passed else 'FAIL'} ({_elapsed_time:.2f}s)"
            for (_test_label, _test_passed, _elapsed_time) in results
        },
        message=f"{num_tests_passed}/{total_tests} scripts passed in {total_elapsed_time:.2f}s.",
    )
    if num_tests_passed < total_tests:
        sys.exit(1)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
