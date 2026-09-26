"""Environment construction and the single-agent simulation loop."""

from dataclasses import dataclass

import numpy as np

from simulation.agent import Agent
from simulation.config import Scenario
from simulation.control import ControlModel
from simulation.landscape import NKLandscape
from simulation.tech_tree import TechTree


@dataclass(frozen=True)
class Environment:
    """Sampled once per replication and shared by all intelligence levels."""
    landscape: NKLandscape
    tech_tree: TechTree
    control: ControlModel
    initial_known: set[int]
    noise_dims: set[int]


@dataclass(frozen=True)
class RunResult:
    initial_fitness: float
    final_fitness: float
    time_to_final_fitness: float
    total_confirmed: int
    total_refuted: int
    trajectory: list[tuple[float, float]]  # (time, fitness) at t=0 and after each confirmation


def build_environment(sc: Scenario, rng: np.random.Generator) -> Environment:
    landscape = NKLandscape(sc.N, sc.K_NK, rng)
    tech_tree = TechTree(sc.N, sc.dag_density, rng)
    initial_known = tech_tree.initial_known_set(sc.kappa_0, rng)
    control = ControlModel(sc.tau_test, sc.tau_auth, sc.n_choke, tech_tree)
    noise_dims = set(range(sc.N, sc.N + round(sc.noise_ratio * sc.N)))
    return Environment(landscape, tech_tree, control, initial_known, noise_dims)


def run_single(env: Environment, beta: float, rng: np.random.Generator) -> RunResult:
    """Test hypotheses until every reachable real dimension has been tested.

    Remaining noise hypotheses cannot improve fitness, so the run stops once the real frontier is empty.
    """
    agent = Agent(beta, env.initial_known, env.landscape, rng)
    t = 0.0
    trajectory = [(t, agent.fitness)]
    while frontier := env.tech_tree.frontier(agent.known, agent.refuted):
        dim = agent.select(frontier | (env.noise_dims - agent.refuted), env.control.cost)
        t += env.control.cost(dim)
        if agent.test(dim):
            trajectory.append((t, agent.fitness))
    return RunResult(
        initial_fitness=trajectory[0][1],
        final_fitness=agent.fitness,
        time_to_final_fitness=trajectory[-1][0],
        total_confirmed=len(agent.known) - len(env.initial_known),
        total_refuted=len(agent.refuted),
        trajectory=trajectory,
    )
