import numpy as np

from utils import generalized_gini, regular_gini


class Metric():

    def __init__(self):
        self._value = 0
        # If metric for each agent or not
        self.__agent_based = False

    @property
    def value(self):
        return self._value

    def reset(self):
        self._value = 0

    def update(self, info):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class GiniMetric(Metric):
    """Base class for Gini metrics."""

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.income = np.zeros(self.n_agents)

    def reset(self):
        super().reset()
        self.income = np.zeros(self.n_agents)

    def update(self, info):
        raise NotImplementedError


class GiniRewards(GiniMetric):
    """Generalised Gini coefficient based on rewards

    Generalization from Rafinetti, Siletti and Vernizzi (2015)
    https://doi.org/10.1007/s10260-014-0293-4
    """

    def update(self, info):
        self.income += [info[f"reward{i}"] for i in range(self.n_agents)]
        self._value = generalized_gini(self.income)


class GiniApples(GiniMetric):
    """Gini coefficient based on times agent has picked apples."""

    def update(self, info):
        # Income increases by one for each apple picked
        attempted_pick = np.array([(info[f"action{i}"] == 1) for i in range(self.n_agents)])
        if any(attempted_pick):
            self.income[attempted_pick] += (info["n_picked_apples"] / sum(attempted_pick))
        self._value = regular_gini(self.income)


class GiniTagged(GiniMetric):
    """Gini coefficient over distribution of time agents are tagged out."""

    def update(self, info):
        # Income is number of time steps agent was tagged out.
        self.income += [info[f"was_tagged{i_agent}"] for i_agent in range(self.n_agents)]
        self._value = regular_gini(self.income)


class Efficiency(Metric):
    """Apples picked compared to highest sustainable pick rate."""

    def __init__(self, growth_rate, max_apples, t_max):
        super().__init__()
        self.maximum_rate = growth_rate * max_apples / 4
        self.t_max = t_max
        self.n_picked_apples = 0

    def reset(self):
        super().reset()
        self.n_picked_apples = 0

    def update(self, info):
        self.n_picked_apples += info["n_picked_apples"]
        self._value = self.n_picked_apples / (self.maximum_rate * self.t_max)


class Aggressiveness(Metric):
    """Measure of aggressiveness computed as the fraction of free actions that are tagging."""

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.n_tagging_actions = 0
        self.n_free_actions = 0

    def reset(self):
        super().reset()
        self.n_tagging_actions = 0
        self.n_free_actions = 0

    def update(self, info):
        for i_agent in range(self.n_agents):
            was_free = not info[f"was_tagged{i_agent}"]
            if was_free:
                self.n_tagging_actions += (info[f"action{i_agent}"] > 1)
                self.n_free_actions += 1
        self._value = self.n_tagging_actions / self.n_free_actions


class SelfHarm(Metric):
    """Measures the fraction of free actions that are punishing one-self."""

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.n_tagging_self = 0
        self.n_free_actions = 0

    def reset(self):
        super().reset()
        self.n_tagging_self = 0
        self.n_free_actions = 0

    def update(self, info):
        for i_agent in range(self.n_agents):
            was_free = not info[f"was_tagged{i_agent}"]
            if was_free:
                self.n_tagging_self += ((info[f"action{i_agent}"] - 2) == i_agent)
                self.n_free_actions += 1
        self._value = self.n_tagging_self / self.n_free_actions


if __name__ == "__main__":
    # Here I put some tests

    gen_gini = generalized_gini
    gini = regular_gini

    print("Generalised Gini vs normal Gini:")
    income = np.ones(10)
    print("Perfect equality", gen_gini(income), gini(income))
    income = np.zeros(10);
    income[np.random.choice(10)] = np.random.rand()
    print("Perfect inequality", gen_gini(income), gini(income))
    income = np.random.rand(10)
    print("Positive income", gen_gini(income), gini(income))
    income = np.random.randn(10)
    print("Zero average income", gen_gini(income), gini(income))
    income = -np.random.rand(10)
    print("All negative income", gen_gini(income), gini(income))
    income = -np.ones(10)
    print("All negative income", gen_gini(income), gini(income))
    income = np.zeros(10)
    income[np.random.choice(10)] = -np.random.rand()
    print("Negative inequality", gen_gini(income), gini(income))
    income = np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, 45.01])
    print("Case (a)", gen_gini(income), gini(income))
    income = np.array([-45, 0, 0, 0, 0, 0, 0, 0, 0, 45.01])
    print("Case (b)", gen_gini(income), gini(income))
    income = np.array([-15, -10, -8, -7, -5, 0, 0, 0, 0, 45.01])
    print("Case (c)", gen_gini(income), gini(income))
