"""Control restrictions: chokepoint dimensions whose tests incur an authorization delay."""

from simulation.tech_tree import TechTree


class ControlModel:
    """Places n_choke chokepoints on the highest-fan-out DAG nodes. Their tests cost tau_auth * tau_test."""

    def __init__(self, tau_test: float, tau_auth: float, n_choke: int, tech_tree: TechTree):
        self.tau_test = tau_test
        self.tau_auth = tau_auth
        self.chokepoints: set[int] = set(tech_tree.gateway_nodes(n_choke)) if n_choke else set()

    def cost(self, dim: int) -> float:
        return self.tau_auth * self.tau_test if dim in self.chokepoints else self.tau_test
