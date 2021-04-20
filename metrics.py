import numpy as np

from utils import generalized_gini, regular_gini

class Metric():
    
    def __init__(self):
        self.__value = 0
        # If metric for each agent or not
        self.__agent_based = False

    @property
    def value(self):
        return self.__value

    def reset(self):
        pass

    def update(self, info):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class GiniRewards(Metric):
    '''Generalised Gini coefficient based on rewards
    
        Generalization from Rafinetti, Siletti and Vernizzi
       https://doi.org/10.1007/s10260-014-0293-4
    '''

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.income = np.zeros(self.n_agents)

    def reset(self):
        self.income = np.zeros(self.n_agents)

    def update(self, info):
        self.income += [info[f"reward{i}"] for i in range(self.n_agents)]
        self.__value = generalized_gini(self.income)

class GiniApples(Metric):
    '''Gini coefficient based on times agent has picked apples.'''

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.income = np.zeros(self.n_agents)

    def reset(self):
        self.income = np.zeros(self.n_agents)

    def update(self, info):
        # Income increases by one for each attempted apple pick
        self.income += [info[f"action{i}"]==1 for i in range(self.n_agents)]
        self.__value = regular_gini(self.income)



if __name__=="__main__":
    # Here I put some tests

    gen_gini = generalized_gini
    gini = regular_gini

    print("Generalised Gini vs normal Gini:")
    income = np.ones(10)
    print("Perfect equality", gen_gini(income), gini(income))
    income = np.zeros(10); income[np.random.choice(10)] = np.random.rand()
    print("Perfect inequality", gen_gini(income), gini(income))
    income = np.random.rand(10)
    print("Positive income", gen_gini(income), gini(income))
    income = np.random.randn(10)
    print("Zero average income", gen_gini(income), gini(income))
    income = -np.random.rand(10)
    print("All negative income", gen_gini(income), gini(income))
    income = -np.ones(10)
    print("All negative income", gen_gini(income), gini(income))
    income = np.zeros(10); income[np.random.choice(10)] = -np.random.rand()
    print("Negative inequality", gen_gini(income), gini(income))
    income = np.array([-5, -5, -5, -5, -5, -5, -5 ,-5, -5, 45.01])
    print("Case (a)", gen_gini(income), gini(income))
    income = np.array([-45, 0, 0, 0, 0, 0, 0 ,0, 0, 45.01])
    print("Case (b)", gen_gini(income), gini(income))
    income = np.array([-15, -10, -8, -7, -5, 0, 0 ,0, 0, 45.01])
    print("Case (c)", gen_gini(income), gini(income))
    
