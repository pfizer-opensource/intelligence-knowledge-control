"""Table 3 of the paper and the statistics quoted in the text, computed from the stored runs."""

import json
from pathlib import Path

import pandas as pd

from simulation.config import (
    BASELINES, BETA_ORDER, INTELLIGENCE_LEVELS, ORACLE, PRIMARY_SEED, RUNS_FILE, SCENARIOS, SEEDS,
    STATISTICS_FILE, SUPER, TABLE_FILE, TIME as T,
)


def _by_cell(runs: pd.DataFrame, column: str, agg: str) -> pd.DataFrame:
    """Aggregate *column* into a scenario x intelligence-level frame in canonical order."""
    return runs.groupby(["scenario", "beta_label"])[column].agg(agg).unstack().loc[list(SCENARIOS), BETA_ORDER]


def _multipliers(mean_t: pd.DataFrame) -> pd.DataFrame:
    return mean_t[BASELINES].div(mean_t[SUPER], axis=0)


def _fmt_time(v: float) -> str:
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def latex_table(mean_t: pd.DataFrame) -> str:
    mult = _multipliers(mean_t)
    betas = dict(INTELLIGENCE_LEVELS)
    beta_cells = " & ".join(rf"$\beta\!=\!{betas[bl]}$" for bl in BETA_ORDER)
    lines = [
        r"\begin{tabular}{l rrrrr ccc}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Mean time to final fitness $\bar{T}$} & \multicolumn{3}{c}{Multiplier vs.\ super ($\beta\!=\!0.9$)} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-9}",
        r"Scenario & Random & Naive & Regular & Super & Oracle & Random & Naive & Regular \\",
        rf" & {beta_cells} & & & \\",
        r"\midrule",
    ]
    for name, sc in SCENARIOS.items():
        times = " & ".join(_fmt_time(mean_t.loc[name, bl]) for bl in BETA_ORDER)
        mults = " & ".join(rf"${mult.loc[name, bl]:.1f}\times$" for bl in BASELINES)
        lines.append(rf"{sc.title} & {times} & {mults} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _jsonable(v: pd.DataFrame | pd.Series | float) -> dict | float:
    if isinstance(v, pd.DataFrame):
        return v.round(3).to_dict(orient="index")
    if isinstance(v, pd.Series):
        return v.round(3).to_dict()
    return round(float(v), 3)


def statistics(runs_by_seed: dict[int, pd.DataFrame]) -> dict:
    runs = runs_by_seed[PRIMARY_SEED]
    mean_t = _by_cell(runs, T, "mean")
    cv = _by_cell(runs, T, "std") / mean_t
    regular_mult = pd.DataFrame({seed: _multipliers(_by_cell(r, T, "mean"))["regular"]
                                 for seed, r in runs_by_seed.items()})
    stats = {
        "mean_time_to_final_fitness": mean_t,
        "multiplier_vs_super": _multipliers(mean_t),
        "oracle_boost_super_over_oracle": mean_t[SUPER] / mean_t[ORACLE],
        "coefficient_of_variation": cv[["random", ORACLE]],
        "mean_confirmed_hypotheses": _by_cell(runs, "total_confirmed", "mean"),
        "chokepoint_slowdown_digital_2b_over_2a": mean_t.loc["2b_digital_strong"] / mean_t.loc["2a_digital_weak"],
        "chokepoint_slowdown_physical_3b_over_3a": mean_t.loc["3b_physical_strong"] / mean_t.loc["3a_physical_weak"],
        "oracle_physical_over_digital_3a_over_2a": mean_t.loc["3a_physical_weak", ORACLE] / mean_t.loc["2a_digital_weak", ORACLE],
        "regular_multiplier_by_seed": regular_mult,
        "regular_multiplier_sd_across_seeds": regular_mult.std(axis=1),
    }
    return {"primary_seed": PRIMARY_SEED, "seeds": list(runs_by_seed)} | {k: _jsonable(v) for k, v in stats.items()}


def write_report(results_dir: Path) -> None:
    runs_by_seed = {seed: pd.read_csv(results_dir / f"seed_{seed}" / RUNS_FILE)
                    for seed in SEEDS if (results_dir / f"seed_{seed}" / RUNS_FILE).exists()}
    (results_dir / TABLE_FILE).write_text(latex_table(_by_cell(runs_by_seed[PRIMARY_SEED], T, "mean")))
    (results_dir / STATISTICS_FILE).write_text(json.dumps(statistics(runs_by_seed), indent=2) + "\n")
    print(f"Table and statistics written to {results_dir}")
