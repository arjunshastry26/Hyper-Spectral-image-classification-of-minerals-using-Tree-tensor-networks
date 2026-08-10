"""
================================================================================
MASTER PIPELINE RUNNER
Capstone Team 88 — Hyperspectral Mineral Classification with Tree Tensor Networks
================================================================================

Runs all notebooks in the correct dependency order using jupyter nbconvert.

ORDER:
  Step 1 — GMM+SAM Preprocessing Pipeline  (creates patches_train/test.npy)
  Step 2 — Binary Tree TTN
  Step 3 — Ternary Tree TTN
  Step 4 — 4-ary Tree TTN
  Step 5 — MPS / Tensor Train
  Step 6 — Tensor Ring

USAGE:
  python run_pipeline.py                  # run all steps
  python run_pipeline.py --skip-preprocess # skip Step 1 (if patches already saved)
  python run_pipeline.py --only 2 3       # run only Binary and Ternary
  python run_pipeline.py --from 3         # run from Ternary onwards

REQUIREMENTS:
  pip install nbconvert nbclient
================================================================================
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime, timedelta

# ── UPDATE THESE PATHS ────────────────────────────────────────────────────────
NOTEBOOKS_DIR = r"C:\Users\PESU-RF\Desktop\Capstone Team - 88\notebooks"
LOG_DIR       = r"C:\Users\PESU-RF\Desktop\Capstone Team - 88\pipeline_logs"
# ─────────────────────────────────────────────────────────────────────────────

# Pipeline steps — (step_number, display_name, notebook_filename)
PIPELINE = [
    (1, "GMM+SAM Preprocessing",  "GMM__SAM_PIPELINE.ipynb"),
    (2, "Binary Tree TTN",         "TTN_Classifier___GMM___Binary_tree.ipynb"),
    (3, "Ternary Tree TTN",        "TTN_Classifier_Ternary_Tree.ipynb"),
    (4, "4-ary Tree TTN",          "TTN_Classifier_4ary_Tree.ipynb"),
    (5, "MPS / Tensor Train",      "Classifier_MPS.ipynb"),
    (6, "Tensor Ring",             "Classifier_TensorRing.ipynb"),
]

# Step dependencies — step N requires these steps to have run first
DEPENDENCIES = {
    2: [1],  # Binary tree needs preprocessing
    3: [1],  # Ternary tree needs preprocessing
    4: [1],  # 4-ary tree needs preprocessing
    5: [1],  # MPS needs preprocessing
    6: [1],  # Tensor Ring needs preprocessing
}


def separator(char="=", width=70):
    return char * width


def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def run_notebook(nb_path, log_path, timeout_hours=24):
    """
    Execute a notebook using jupyter nbconvert.
    Returns (success: bool, elapsed_seconds: float)
    """
    timeout_s = timeout_hours * 3600
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",                      # overwrite with outputs
        f"--ExecutePreprocessor.timeout={timeout_s}",
        "--ExecutePreprocessor.kernel_name=python3",
        nb_path
    ]

    t0 = time.time()
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_s + 60
        )
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def check_notebook_exists(nb_path):
    if not os.path.exists(nb_path):
        print(f"  ✗ NOT FOUND: {nb_path}")
        return False
    return True


def print_status_table(results):
    print(f"\n{separator()}")
    print("  PIPELINE RESULTS SUMMARY")
    print(separator())
    print(f"  {'#':<4} {'Notebook':<35} {'Status':<10} {'Time':>10}")
    print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*10}")
    for step, name, status, elapsed in results:
        icon    = "✓" if status == "DONE" else ("✗" if status == "FAILED" else "○")
        time_s  = fmt_time(elapsed) if elapsed else "-"
        print(f"  {icon} {step:<3} {name:<35} {status:<10} {time_s:>10}")
    print(separator())


def main():
    parser = argparse.ArgumentParser(description="Run all pipeline notebooks in order.")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip Step 1 (GMM+SAM preprocessing)")
    parser.add_argument("--only", nargs="+", type=int,
                        help="Run only these step numbers e.g. --only 2 3")
    parser.add_argument("--from", dest="from_step", type=int, default=None,
                        help="Start from this step number e.g. --from 3")
    parser.add_argument("--timeout-hours", type=int, default=24,
                        help="Timeout per notebook in hours (default: 24)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # Determine which steps to run
    steps_to_run = [s for s, _, _ in PIPELINE]

    if args.skip_preprocess:
        steps_to_run = [s for s in steps_to_run if s != 1]

    if args.only:
        steps_to_run = [s for s in steps_to_run if s in args.only]

    if args.from_step:
        steps_to_run = [s for s in steps_to_run if s >= args.from_step]

    # Filter pipeline to selected steps
    selected = [(s, n, f) for s, n, f in PIPELINE if s in steps_to_run]

    print(separator())
    print("  CAPSTONE TEAM 88 — MASTER PIPELINE RUNNER")
    print(separator())
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps      : {[s for s, _, _ in selected]}")
    print(f"  Notebooks  : {NOTEBOOKS_DIR}")
    print(f"  Logs       : {LOG_DIR}")
    print(f"  Timeout    : {args.timeout_hours}h per notebook")
    print(separator())

    results = []
    pipeline_start = time.time()

    for step, name, filename in selected:
        nb_path  = os.path.join(NOTEBOOKS_DIR, filename)
        log_path = os.path.join(LOG_DIR, f"step{step:02d}_{filename.replace('.ipynb', '')}.log")

        print(f"\n{separator('-')}")
        print(f"  STEP {step}/{len(PIPELINE)} — {name}")
        print(f"  Notebook : {filename}")
        print(f"  Log      : {os.path.basename(log_path)}")
        print(f"  Started  : {datetime.now().strftime('%H:%M:%S')}")
        print(separator("-"))

        # Check notebook exists
        if not check_notebook_exists(nb_path):
            print(f"  ✗ SKIPPED — notebook not found")
            results.append((step, name, "SKIPPED", None))
            continue

        # Check dependencies
        deps    = DEPENDENCIES.get(step, [])
        dep_ok  = all(
            any(r[0] == d and r[2] == "DONE" for r in results)
            for d in deps
        )
        if deps and not dep_ok:
            print(f"  ✗ SKIPPED — dependencies not met (needs steps {deps})")
            results.append((step, name, "SKIPPED", None))
            continue

        # Run notebook
        print(f"  Running... (this may take several hours)")
        print(f"  Progress output → {os.path.basename(log_path)}")

        try:
            success, elapsed = run_notebook(nb_path, log_path, args.timeout_hours)

            if success:
                print(f"  ✓ DONE in {fmt_time(elapsed)}")
                results.append((step, name, "DONE", elapsed))
            else:
                print(f"  ✗ FAILED after {fmt_time(elapsed)}")
                print(f"    Check log: {log_path}")
                results.append((step, name, "FAILED", elapsed))

                # Ask whether to continue
                try:
                    ans = input("\n  Continue with next step? [y/N]: ").strip().lower()
                    if ans != "y":
                        print("  Pipeline stopped by user.")
                        break
                except EOFError:
                    print("  Non-interactive mode — stopping on failure.")
                    break

        except subprocess.TimeoutExpired:
            elapsed = args.timeout_hours * 3600
            print(f"  ✗ TIMEOUT after {args.timeout_hours}h")
            results.append((step, name, "TIMEOUT", elapsed))

        except KeyboardInterrupt:
            print(f"\n  Pipeline interrupted by user.")
            results.append((step, name, "INTERRUPTED", None))
            break

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append((step, name, "ERROR", None))

    # Final summary
    total_elapsed = time.time() - pipeline_start
    print_status_table(results)
    print(f"  Total time : {fmt_time(total_elapsed)}")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(separator())

    # Write summary to log
    summary_path = os.path.join(LOG_DIR, "pipeline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Pipeline run: {datetime.now()}\n")
        f.write(f"Total time : {fmt_time(total_elapsed)}\n\n")
        for step, name, status, elapsed in results:
            t = fmt_time(elapsed) if elapsed else "-"
            f.write(f"Step {step}: {name:<35} {status:<10} {t}\n")
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
