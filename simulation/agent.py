"""Epistemic agent that selects and tests hypotheses.

Real dimensions 0..N-1 are landscape loci. Noise dimensions N..N+M-1 never improve fitness.
A test confirms a real dimension if flipping it increases fitness and refutes it otherwise.
"""

from collections.abc import Callable

import numpy as np

from simulation.landscape import NKLandscape


class Agent:
    """With probability beta the agent picks the candidate with the highest fitness gain per unit cost,
    otherwise it picks a candidate uniformly at random."""

    def __init__(self, beta: float, initial_known: set[int], landscape: NKLandscape,
                 rng: np.random.Generator):
        self.beta = beta
        self.landscape = landscape
        self.rng = rng
        self.position = np.zeros(landscape.N, dtype=np.int8)
        self.position[list(initial_known)] = 1
        self.known: set[int] = set(initial_known)
        self.refuted: set[int] = set()
        self.fitness = landscape.fitness(self.position)

    def _delta(self, dim: int) -> float:
        return 0.0 if dim >= self.landscape.N else self.landscape.delta(self.position, dim)

    def select(self, pool: set[int], cost: Callable[[int], float]) -> int:
        candidates = list(pool)
        if self.rng.random() < self.beta:
            return max(candidates, key=lambda d: self._delta(d) / cost(d))
        return self.rng.choice(candidates)

    def test(self, dim: int) -> bool:
        """Test *dim*. Returns True if confirmed."""
        if dim < self.landscape.N and self.landscape.delta(self.position, dim) > 0:
            self.position[dim] = 1
            self.known.add(dim)
            self.fitness = self.landscape.fitness(self.position)
            return True
        self.refuted.add(dim)
        return False
