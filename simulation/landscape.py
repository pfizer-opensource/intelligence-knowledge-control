"""NK fitness landscape (Kauffman, 1993) over the real hypothesis dimensions."""

import numpy as np

from simulation.config import OPTIMUM_SEARCH_STARTS


class NKLandscape:
    """Each locus contributes a value that depends on its own allele and on K_NK partner loci."""

    def __init__(self, N: int, K_NK: int, rng: np.random.Generator):
        self.N = N
        self.K_NK = min(K_NK, N - 1)

        self.partners: list[list[int]] = []
        for i in range(N):
            others = np.array([j for j in range(N) if j != i])
            self.partners.append(sorted(rng.choice(others, size=self.K_NK, replace=False).tolist()))

        # Table index: most significant bit = allele at the locus, then partner alleles in order.
        self.tables: list[np.ndarray] = [rng.random(1 << (self.K_NK + 1)) for _ in range(N)]

        # Loci whose contribution changes when a given dimension is flipped.
        self._affected_by: list[list[int]] = [[] for _ in range(N)]
        for i in range(N):
            self._affected_by[i].append(i)
            for partner in self.partners[i]:
                self._affected_by[partner].append(i)

        self.max_fitness = self._estimate_max_fitness(rng)

    def _contribution(self, position: np.ndarray, locus: int) -> float:
        idx = int(position[locus])
        for partner in self.partners[locus]:
            idx = (idx << 1) | int(position[partner])
        return self.tables[locus][idx]

    def fitness(self, position: np.ndarray) -> float:
        # Plain accumulation: built-in sum() uses compensated summation since Python 3.12.
        total = 0.0
        for i in range(self.N):
            total += self._contribution(position, i)
        return total / self.N

    def delta(self, position: np.ndarray, dim: int) -> float:
        """Fitness change from flipping *dim*, recomputing only the affected loci."""
        affected = self._affected_by[dim]
        old = 0.0
        for loc in affected:
            old += self._contribution(position, loc)
        position[dim] = 1 - position[dim]
        new = 0.0
        for loc in affected:
            new += self._contribution(position, loc)
        position[dim] = 1 - position[dim]
        return (new - old) / self.N

    def _estimate_max_fitness(self, rng: np.random.Generator) -> float:
        """Best local optimum found by multi-start steepest-ascent hill climbing (diagnostic only)."""
        best = -np.inf
        for _ in range(OPTIMUM_SEARCH_STARTS):
            pos = rng.integers(0, 2, size=self.N, dtype=np.int8)
            fit = self.fitness(pos)
            while True:
                best_delta, best_dim = 0.0, -1
                for d in range(self.N):
                    delta = self.delta(pos, d)
                    if delta > best_delta:
                        best_delta, best_dim = delta, d
                if best_dim < 0:
                    break
                pos[best_dim] = 1 - pos[best_dim]
                fit += best_delta
            best = max(best, fit)
        return best
