# Simulation of Epistemic Exploration

Code and results for the computational stress-test in the NeurIPS 2026 position paper
*AI Governance Should Prioritize Control and Knowledge Boundaries Over Limiting Intelligence*.

An epistemic agent searches an NK fitness landscape (Kauffman, 1993) by testing hypotheses. Each test costs time whether or not the hypothesis is confirmed. The simulation extends the NK landscape with noise dimensions, dependencies between hypotheses, and test costs. The model elements map onto the concepts of the paper as follows:

- **Hypotheses about the world.** Each dimension corresponds to a hypothesis. Value 1 denotes a confirmed hypothesis. Value 0 denotes a hypothesis that has not yet been tested or has been refuted, and refuted hypotheses are recorded separately. A test confirms a hypothesis if setting its value to 1 yields a fitness gain > 0 given the currently confirmed hypotheses and refutes it otherwise. Real landscape dimensions are potentially productive hypotheses, whose outcome depends on the current knowledge state through epistatic interactions. Noise dimensions are constitutionally unproductive hypotheses, which never yield a fitness gain. The agent does not know which hypotheses correspond to noise dimensions, because both kinds of hypothesis enter the same pool of candidates and are selected and tested by the same rule.
- **Prerequisite knowledge (technology tree).** A random acyclic graph over the real dimensions defines dependencies between hypotheses. A real dimension becomes testable only after all its prerequisite dimensions are confirmed. Noise dimensions have no prerequisites.
- **Knowledge cutoff $\mathcal{K}_0$.** The agent starts with a fraction `kappa_0` of the real dimensions confirmed, chosen consistently with the prerequisites.
- **Epistemic intelligence (Definition 1).** With probability `beta` the agent selects the testable hypothesis with the highest fitness gain per unit cost, and otherwise a testable hypothesis uniformly at random.
- **Experimental latency.** Each test costs `tau_test`.
- **Control restrictions.** The `n_choke` real dimensions with the most dependent dimensions are chokepoints, and each test of a chokepoint costs `tau_auth * tau_test`.

The six scenarios and five intelligence levels are defined in [`simulation/config.py`](simulation/config.py).
The primary endpoint is the time to final fitness, i.e. the time of the last confirmed hypothesis.

## Requirements

Python >= 3.12 and the pinned packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

All commands are run from the repository root.

```bash
# Regenerate figures, Table 3, and summary statistics from the stored results (seconds)
python -m simulation.main --analysis-only

# Full reproduction: 4 master seeds x 6 scenarios x 100 replications x 5 intelligence levels.
# Seeds run in parallel, one process each (about 15 minutes on a laptop).
python -m simulation.main

# Quick test (3 replications, one seed). Output goes to smoke/, which is git-ignored.
python -m simulation.main --seeds 42 -n 3 --results-dir smoke/results --figures-dir smoke/figures
```

Each master seed initializes a single random stream from which all environments and agent choices are drawn.
Within a replication, all five intelligence levels are evaluated on the same sampled environment
(landscape, DAG, chokepoints, and initial knowledge set).
The full reproduction regenerates the stored CSV files exactly.

## Outputs

| File | Content | Paper |
|---|---|---|
| `results/seed_<s>/all_runs.csv` | One row per run (scenario, replication, intelligence level) | Table 3, Figures 3-6, seed robustness (Appendix H) |
| `results/seed_<s>/trajectories.csv` | Fitness at t = 0 and after each confirmation | Figures 7, 8 |
| `results/table_multiplier_summary.tex` | Mean time to final fitness and intelligence multipliers (seed 42) | Table 3 |
| `results/statistics.json` | Statistics quoted in the text (multipliers, oracle boost, CVs, chokepoint slowdowns, seed SDs) | Section 5, Appendices G and H |
| `figures/fig2_intelligence_multiplier.pdf` | Intelligence multiplier with 95% bootstrap CIs | Figure 3 |
| `figures/fig5_oracle_sanity.pdf` | Speedup from beta = 0.9 to beta = 1.0 | Figure 4 |
| `figures/fig1_scenario_comparison.pdf` | Time to final fitness by scenario and intelligence level | Figure 5 |
| `figures/fig3_chokepoint_effect.pdf` | Mean time to final fitness with and without chokepoints | Figure 6 |
| `figures/fig4_trajectories.pdf` | Median fitness trajectories with interquartile ranges | Figure 7 |
| `figures/fig6_fitness_dynamics.pdf` | Mean normalized fitness gain over time | Figure 8 |

Each row of `all_runs.csv` describes one run, i.e. one intelligence level in one replication of one scenario. The columns are:

- `scenario`: name of the scenario in `SCENARIOS` in [`simulation/config.py`](simulation/config.py).
- `replication`: index of the replication. All intelligence levels within a replication share the same landscape, dependencies between hypotheses, initially confirmed hypotheses, and chokepoints.
- `beta_label`, `beta`: name and selection bias of the intelligence level.
- Simulation parameters:
  - `N`: the number of real dimensions
  - `K_NK`: the number of epistatic partners per real dimension
  - `dag_density`: the edge probability of the dependency graph
  - `tau_test`: the cost of one test
  - `noise_ratio`: the number of noise dimensions per real dimension
  - `kappa_0`: the initial fraction of confirmed real hypotheses
  - `n_choke`: the number of chokepoints
  - `tau_auth`: the cost multiplier for tests of chokepoints
- `initial_fitness`: fitness conferred by the hypotheses confirmed under `kappa_0`.
- `landscape_max`: highest local optimum found by hill climbing from 50 random starting positions. This column is a diagnostic and does not enter any reported result.
- `final_fitness`: fitness at the end of the run.
- `time_to_final_fitness`: primary endpoint, the cumulative test cost up to the last confirmed hypothesis.
- `total_confirmed`: number of hypotheses confirmed during the run, excluding the hypotheses confirmed under `kappa_0`.
- `total_refuted`: number of hypotheses refuted during the run

## Code structure

```
simulation/
├── config.py       Scenarios, intelligence levels, and all constants
├── landscape.py    NK fitness landscape
├── tech_tree.py    Prerequisite DAG
├── control.py      Chokepoints and authorization delays
├── agent.py        Hypothesis selection and testing
├── simulation.py   Environment construction and simulation loop
├── plotting.py     Figures
├── report.py       Table 3 and summary statistics
└── main.py         Command-line entry point
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
