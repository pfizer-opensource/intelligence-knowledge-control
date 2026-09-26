"""Scenario registry, intelligence levels, and all shared constants."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    label: str          # Short label used in figures ("0", "1", "2A", ...)
    title: str          # Row label in the summary table
    N: int              # Number of real hypothesis dimensions
    K_NK: int           # NK epistatic coupling (landscape ruggedness)
    dag_density: float  # Edge probability of the prerequisite DAG
    tau_test: float     # Time cost of one hypothesis test
    noise_ratio: float  # Noise hypotheses per real dimension
    kappa_0: float      # Initial knowledge fraction
    n_choke: int = 0    # Number of chokepoint dimensions
    tau_auth: float = 1.0  # Test-cost multiplier for chokepoint dimensions


# Insertion order is part of the reproducible random stream.
SCENARIOS: dict[str, Scenario] = {
    "0_trivial": Scenario("0", "0: Connecting the dots", N=100, K_NK=2, dag_density=0.01,
                          tau_test=1.0, noise_ratio=0.0, kappa_0=0.70),
    "1_disaster": Scenario("1", "1: Disaster", N=100, K_NK=2, dag_density=0.01,
                           tau_test=1.0, noise_ratio=3.0, kappa_0=0.70),
    "2a_digital_weak": Scenario("2A", "2A: Digital", N=100, K_NK=4, dag_density=0.05,
                                tau_test=1.0, noise_ratio=9.0, kappa_0=0.10),
    "2b_digital_strong": Scenario("2B", "2B: Dig.+choke", N=100, K_NK=4, dag_density=0.05,
                                  tau_test=1.0, noise_ratio=9.0, kappa_0=0.10,
                                  n_choke=10, tau_auth=20.0),
    "3a_physical_weak": Scenario("3A", "3A: Physical", N=100, K_NK=6, dag_density=0.05,
                                 tau_test=100.0, noise_ratio=9.0, kappa_0=0.10),
    "3b_physical_strong": Scenario("3B", "3B: Phys.+choke", N=100, K_NK=6, dag_density=0.05,
                                   tau_test=100.0, noise_ratio=9.0, kappa_0=0.10,
                                   n_choke=10, tau_auth=5.0),
}

# (label, beta). Run order within a replication is part of the reproducible random stream.
INTELLIGENCE_LEVELS: list[tuple[str, float]] = [
    ("regular", 0.1),
    ("superintelligent", 0.9),
    ("oracle", 1.0),
    ("naive", 0.009),
    ("random", 0.0009),
]
BETA_ORDER: list[str] = [label for label, _ in sorted(INTELLIGENCE_LEVELS, key=lambda lv: lv[1])]
SUPER = "superintelligent"
ORACLE = "oracle"
BASELINES: list[str] = ["random", "naive", "regular"]

N_REPLICATIONS = 100
SEEDS: tuple[int, ...] = (42, 0, 7, 2024)
PRIMARY_SEED = 42
OPTIMUM_SEARCH_STARTS = 50
BOOTSTRAP_SAMPLES = 1000

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
RUNS_FILE = "all_runs.csv"
TRAJECTORIES_FILE = "trajectories.csv"
TABLE_FILE = "table_multiplier_summary.tex"
STATISTICS_FILE = "statistics.json"
TIME = "time_to_final_fitness"  # Primary endpoint

# Plotting
BETA_COLORS: dict[str, str] = {
    "random": "#7b1fa2",
    "naive": "#00897b",
    "regular": "#1976d2",
    "superintelligent": "#ff9800",
    "oracle": "#d32f2f",
}
BETA_LABELS: dict[str, str] = {label: rf"$\beta={beta}$ ({label.replace('superintelligent', 'super')})"
                               for label, beta in INTELLIGENCE_LEVELS}
TRAJECTORY_XLIM: dict[str, float] = {
    "0_trivial": 30,
    "1_disaster": 300,
    "2a_digital_weak": 500,
    "2b_digital_strong": 500,
    "3a_physical_weak": 10_000,
    "3b_physical_strong": 10_000,
}
DYNAMICS_SCENARIOS: list[str] = ["0_trivial", "1_disaster", "2a_digital_weak"]
CHOKEPOINT_PAIRS: list[tuple[str, str, str]] = [
    ("2a_digital_weak", "2b_digital_strong", "Digital (2A vs 2B)"),
    ("3a_physical_weak", "3b_physical_strong", "Physical (3A vs 3B)"),
]
