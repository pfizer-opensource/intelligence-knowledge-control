"""Technology tree: prerequisite DAG over the real hypothesis dimensions."""

from collections import defaultdict

import networkx as nx
import numpy as np


class TechTree:
    """Random DAG on a random topological order. A dimension is testable once all its prerequisites are known."""

    def __init__(self, N: int, density: float, rng: np.random.Generator):
        self.N = N
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(N))
        self._rank = {node: i for i, node in enumerate(rng.permutation(N).tolist())}
        for i in range(N):
            for j in range(N):
                if self._rank[i] < self._rank[j] and rng.random() < density:
                    self.dag.add_edge(i, j)
        self._predecessors = {node: frozenset(self.dag.predecessors(node)) for node in range(N)}

    def frontier(self, known: set[int], refuted: set[int]) -> set[int]:
        """Untested dimensions whose prerequisites are all confirmed."""
        excluded = known | refuted
        return {node for node in range(self.N)
                if node not in excluded and self._predecessors[node].issubset(known)}

    def initial_known_set(self, kappa_0: float, rng: np.random.Generator) -> set[int]:
        """Prerequisite-consistent set of round(kappa_0 * N) dimensions, shuffled within DAG depth levels."""
        n_known = round(kappa_0 * self.N)
        depths = {n: 0 for n in range(self.N)}
        by_depth: dict[int, list[int]] = defaultdict(list)
        for n in nx.topological_sort(self.dag):
            for pred in self.dag.predecessors(n):
                depths[n] = max(depths[n], depths[pred] + 1)
            by_depth[depths[n]].append(n)
        known: set[int] = set()
        for d in sorted(by_depth):
            group = by_depth[d]
            rng.shuffle(group)
            for node in group:
                if len(known) < n_known and self._predecessors[node].issubset(known):
                    known.add(node)
        return known

    def gateway_nodes(self, n: int) -> list[int]:
        """The n nodes with the most transitive descendants (ties: earlier topological rank)."""
        desc_count = {node: len(nx.descendants(self.dag, node)) for node in range(self.N)}
        return sorted(range(self.N), key=lambda d: (-desc_count[d], self._rank[d]))[:n]
