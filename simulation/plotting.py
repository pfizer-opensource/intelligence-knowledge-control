"""Figures 3-8 of the paper, generated from the primary-seed results."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from simulation.config import (
    BASELINES, BETA_COLORS, BETA_LABELS, BETA_ORDER, BOOTSTRAP_SAMPLES, CHOKEPOINT_PAIRS,
    DYNAMICS_SCENARIOS, ORACLE, PRIMARY_SEED, RUNS_FILE, SCENARIOS, SUPER, TIME as T, TRAJECTORIES_FILE,
    TRAJECTORY_XLIM,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _times(runs: pd.DataFrame, scenario: str, beta_label: str) -> np.ndarray:
    return runs.loc[(runs["scenario"] == scenario) & (runs["beta_label"] == beta_label), T].to_numpy()


def _bootstrap_ci(num: np.ndarray, den: np.ndarray | None, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap CI of mean(num), or of mean(num) / mean(den) if den is given."""
    stats = []
    for _ in range(BOOTSTRAP_SAMPLES):
        stat = rng.choice(num, size=len(num)).mean()
        if den is not None:
            stat /= rng.choice(den, size=len(den)).mean()
        stats.append(stat)
    return tuple(np.percentile(stats, [2.5, 97.5]))


def _trajectory_matrix(traj: pd.DataFrame, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step-function fitness of each run on t_grid, and each run's final fitness."""
    rows, finals = [], []
    for _, run in traj.groupby("replication"):
        f = run["fitness"].to_numpy()
        rows.append(f[np.searchsorted(run["t"].to_numpy(), t_grid, side="right") - 1])
        finals.append(f[-1])
    return np.array(rows), np.array(finals)


def fig_scenario_comparison(runs: pd.DataFrame) -> plt.Figure:
    """Box plot of time to final fitness per scenario and intelligence level (whiskers: 5th-95th pct)."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(BETA_ORDER)
    for i, bl in enumerate(BETA_ORDER):
        offset = (i - (len(BETA_ORDER) - 1) / 2) * width
        bp = ax.boxplot(
            # Floor keeps zero times finite on the log axis.
            [np.clip(_times(runs, sc, bl), 0.5, None) for sc in SCENARIOS],
            positions=np.arange(len(SCENARIOS)) + offset,
            widths=width * 0.85, patch_artist=True, showfliers=False, whis=[5, 95],
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(BETA_COLORS[bl])
            patch.set_alpha(0.7)
        ax.bar(0, 0, color=BETA_COLORS[bl], alpha=0.7, label=BETA_LABELS[bl])
    ax.set_yscale("log")
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels([sc.label for sc in SCENARIOS.values()], fontsize=8)
    ax.set_ylabel("Time to final fitness")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_intelligence_multiplier(runs: pd.DataFrame, rng: np.random.Generator) -> plt.Figure:
    """Ratio of mean time to final fitness (baseline / superintelligent) with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(BASELINES)
    for i, bl in enumerate(BASELINES):
        mult, err_lo, err_hi = [], [], []
        for sc in SCENARIOS:
            lo, hi = _times(runs, sc, bl), _times(runs, sc, SUPER)
            m = lo.mean() / hi.mean()
            ci = _bootstrap_ci(lo, hi, rng)
            mult.append(m)
            err_lo.append(m - ci[0])
            err_hi.append(ci[1] - m)
        ax.bar(x + (i - (len(BASELINES) - 1) / 2) * width, mult, width=width * 0.9,
               yerr=[err_lo, err_hi], capsize=3, color=BETA_COLORS[bl], alpha=0.85,
               edgecolor="black", linewidth=0.5, label=BETA_LABELS[bl])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([sc.label for sc in SCENARIOS.values()], fontsize=8)
    ax.set_ylabel("Intelligence multiplier at final fitness")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_chokepoint_effect(runs: pd.DataFrame, rng: np.random.Generator) -> plt.Figure:
    """Mean time to final fitness without (plain) and with (hatched) chokepoints, on a broken y-axis."""
    fig = plt.figure(figsize=(9, 4.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.35, height_ratios=[1, 2])
    x = np.arange(len(BETA_ORDER))
    width = 0.35
    axes = []
    for col, (sc_a, sc_b, title) in enumerate(CHOKEPOINT_PAIRS):
        ax_top, ax_bot = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        axes.append((ax_top, ax_bot))
        bars = {}
        for sc in (sc_a, sc_b):
            means, err_lo, err_hi = [], [], []
            for bl in BETA_ORDER:
                vals = _times(runs, sc, bl)
                ci = _bootstrap_ci(vals, None, rng)
                means.append(vals.mean())
                err_lo.append(vals.mean() - ci[0])
                err_hi.append(ci[1] - vals.mean())
            bars[sc] = (means, err_lo, err_hi)

        # Break the axis at the largest gap between sorted means.
        all_means = sorted(set(bars[sc_a][0] + bars[sc_b][0]))
        gap_idx = max(range(len(all_means) - 1), key=lambda i: all_means[i + 1] - all_means[i])
        top = max(m + e for means, _, err_hi in bars.values() for m, e in zip(means, err_hi))

        for ax in (ax_top, ax_bot):
            for sc, offset, hatch in ((sc_a, -width / 2, None), (sc_b, width / 2, "///")):
                means, err_lo, err_hi = bars[sc]
                ax.bar(x + offset, means, width, yerr=[err_lo, err_hi], capsize=3,
                       color=[BETA_COLORS[bl] for bl in BETA_ORDER], edgecolor="black",
                       linewidth=0.5, alpha=0.85, hatch=hatch)
            ax.grid(axis="y", alpha=0.3)
        ax_bot.set_ylim(0, all_means[gap_idx] * 1.5)
        ax_top.set_ylim(all_means[gap_idx + 1] * 0.85, top * 1.08)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_top.set_xticks([])
        ax_top.set_title(title, fontsize=10)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([BETA_LABELS[bl].split("(")[0].strip() for bl in BETA_ORDER], fontsize=8)
        ax_bot.set_ylabel("Mean time to final fitness", fontsize=9)
        ax_top.legend(handles=[
            Patch(facecolor="white", edgecolor="black", linewidth=0.5, label="Baseline"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.5, hatch="///", label="Chokepoints"),
        ], fontsize=7, loc="upper right")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12)
    # Break marks are drawn in figure coordinates so that both diagonals are parallel.
    fig.canvas.draw()
    d = 0.008
    for ax_top, ax_bot in axes:
        top_bb, bot_bb = ax_top.get_position(), ax_bot.get_position()
        for xf in (top_bb.x0, top_bb.x1):
            for y in (top_bb.y0, bot_bb.y1):
                fig.add_artist(plt.Line2D((xf - d, xf + d), (y - d, y + d), color="k", linewidth=0.8,
                                          transform=fig.transFigure, clip_on=False))
    return fig


def fig_trajectories(traj: pd.DataFrame) -> plt.Figure:
    """Median fitness trajectory with interquartile band per scenario and intelligence level."""
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(3.2 * len(SCENARIOS), 3))
    for ax, (name, sc) in zip(axes, SCENARIOS.items()):
        t_grid = np.linspace(0, TRAJECTORY_XLIM[name], 500)
        for bl in BETA_ORDER:
            matrix, _ = _trajectory_matrix(traj[(traj["scenario"] == name) & (traj["beta_label"] == bl)], t_grid)
            ax.plot(t_grid, np.median(matrix, axis=0), label=BETA_LABELS[bl], color=BETA_COLORS[bl],
                    linewidth=1.2, alpha=0.85)
            ax.fill_between(t_grid, *np.percentile(matrix, [25, 75], axis=0), color=BETA_COLORS[bl], alpha=0.15)
        ax.set_xlim(0, TRAJECTORY_XLIM[name])
        ax.set_xlabel("Time")
        ax.set_title(sc.label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fitness")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(BETA_ORDER),
               bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.tight_layout()
    return fig


def fig_oracle_sanity(runs: pd.DataFrame) -> plt.Figure:
    """Ratio of mean time to final fitness, superintelligent / oracle."""
    boosts = [_times(runs, sc, SUPER).mean() / _times(runs, sc, ORACLE).mean() for sc in SCENARIOS]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(len(SCENARIOS))
    ax.bar(x, boosts, color="#9AC4B8", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([sc.label for sc in SCENARIOS.values()], fontsize=7)
    ax.set_ylabel("Speedup (0.9 / 1.0) at final fitness")
    ax.set_ylim(0, max(1.5, max(boosts) * 1.3))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_fitness_dynamics(traj: pd.DataFrame) -> plt.Figure:
    """Mean fitness gain normalized to each run's own final fitness, (f(t) - f(0)) / (f_final - f(0))."""
    fig, axes = plt.subplots(1, len(DYNAMICS_SCENARIOS), figsize=(4.2 * len(DYNAMICS_SCENARIOS), 3.2), sharey=True)
    for ax, name in zip(axes, DYNAMICS_SCENARIOS):
        t_grid = np.linspace(0, TRAJECTORY_XLIM[name], 500)
        for bl in BETA_ORDER:
            matrix, finals = _trajectory_matrix(traj[(traj["scenario"] == name) & (traj["beta_label"] == bl)], t_grid)
            gain = finals - matrix[:, 0]
            gain[gain == 0] = 1.0  # Runs without any confirmation stay at zero.
            norm = (matrix - matrix[:, :1]) / gain[:, None]
            ax.plot(t_grid, norm.mean(axis=0), label=BETA_LABELS[bl], color=BETA_COLORS[bl],
                    linewidth=1.3, alpha=0.85)
        ax.set_xlim(0, TRAJECTORY_XLIM[name])
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Time")
        ax.set_title(SCENARIOS[name].label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Normalized fitness gain")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(BETA_ORDER),
               bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.tight_layout()
    return fig


def generate_all(seed_dir: Path, figures_dir: Path) -> None:
    runs = pd.read_csv(seed_dir / RUNS_FILE)
    traj = pd.read_csv(seed_dir / TRAJECTORIES_FILE)
    rng = np.random.default_rng(PRIMARY_SEED)
    figures = {
        "fig1_scenario_comparison": fig_scenario_comparison(runs),
        "fig2_intelligence_multiplier": fig_intelligence_multiplier(runs, rng),
        "fig3_chokepoint_effect": fig_chokepoint_effect(runs, rng),
        "fig4_trajectories": fig_trajectories(traj),
        "fig5_oracle_sanity": fig_oracle_sanity(runs),
        "fig6_fitness_dynamics": fig_fitness_dynamics(traj),
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.pdf", metadata={"CreationDate": None})
        plt.close(fig)
    print(f"Figures written to {figures_dir}")
