

class Metric():
    
    def __init__(self):
        self.__value = 0

    @property
    def value(self):
        return self.__value

    def reset(self):
        pass

    def update(self, info):
        raise NotImplementedError

class Gini(Metric):

    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.income = np.zeros(self.n_agents)

    def reset(self):
        self.income = np.zeros(self.n_agents)

    def update(self, info):
        self.income += [info[f"reward{i}"] for i in range(self.n_agents)]

        # sum_i sum_j |y_i - y_j| / (2 n sum_i y_i)
        y = self.income
        self.__value = (
            np.abs(np.subtract.outer(y, y)).sum()
            / (2 * self.n_agents * y.sum())
        )
