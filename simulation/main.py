"""Command-line entry point: run all seeds, then generate figures, Table 3, and summary statistics.

    python -m simulation.main                  # full reproduction (all seeds, in parallel)
    python -m simulation.main --analysis-only  # regenerate figures and table from stored results
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.config import (
    FIGURES_DIR, INTELLIGENCE_LEVELS, N_REPLICATIONS, PRIMARY_SEED, RESULTS_DIR, RUNS_FILE, SCENARIOS, SEEDS,
    TIME, TRAJECTORIES_FILE,
)
from simulation.plotting import generate_all
from simulation.report import write_report
from simulation.simulation import build_environment, run_single


def run_seed(seed: int, n_replications: int, out_dir: Path) -> None:
    """Run every scenario x replication x intelligence level from a single random stream."""
    rng = np.random.default_rng(seed)
    runs, trajectories = [], []
    for name, sc in SCENARIOS.items():
        params = {k: v for k, v in asdict(sc).items() if k not in ("label", "title")}
        for rep in range(n_replications):
            env = build_environment(sc, rng)
            for label, beta in INTELLIGENCE_LEVELS:
                r = run_single(env, beta, rng)
                runs.append({"scenario": name, "replication": rep, "beta_label": label, "beta": beta, **params,
                             "initial_fitness": r.initial_fitness, "landscape_max": env.landscape.max_fitness,
                             "final_fitness": r.final_fitness, TIME: r.time_to_final_fitness,
                             "total_confirmed": r.total_confirmed, "total_refuted": r.total_refuted})
                trajectories += [{"scenario": name, "replication": rep, "beta_label": label, "t": t, "fitness": f}
                                 for t, f in r.trajectory]
        print(f"[seed {seed}] {name} done", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(out_dir / RUNS_FILE, index=False)
    pd.DataFrame(trajectories).to_csv(out_dir / TRAJECTORIES_FILE, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="master seeds to run")
    parser.add_argument("-n", "--replications", type=int, default=N_REPLICATIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--analysis-only", action="store_true", help="skip simulation and use stored results")
    args = parser.parse_args()

    if not args.analysis_only:
        with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
            futures = [pool.submit(run_seed, s, args.replications, args.results_dir / f"seed_{s}")
                       for s in args.seeds]
            for f in futures:
                f.result()

    generate_all(args.results_dir / f"seed_{PRIMARY_SEED}", args.figures_dir)
    write_report(args.results_dir)


if __name__ == "__main__":
    main()
