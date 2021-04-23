import sys
import numpy as np
from io import StringIO
from contextlib import closing

from gym import Env, spaces

class SimpleHarvest(Env):
    """Simplified version of the Harvest Game in arXiv:1803.08884.

    This game is no longer a gridworld game.

    Observation:
        Apples available
        Previous actions

    Action:
        WAIT (and let apples grow)
        PICK (apples)
        TAG-N (tag agent N from agents viewpoint)

    Reward:
        WAIT: 0
        PICK: +1 when there is enough apples, otherwise share
            with other agents
        TAG: -tag_cost

    Example:
        3 agents, (agent0, agent1, agent2)
        initially 50 apples (half the carrying capacity)
        growth_rate = 0.0  # no new apples will appear
        action history:
            0, 1, 3
            4, 1, 1
            4, 0, 3
        next observations:
        (
            # agent0
            np.array([
                47,  # apples picked
                4,   # action agent0
                0,   # action agent0
                3,   # action agent0
            ]),
            # agent1
            np.array([
                47,  # apples picked
                3,   # action agent0
                4,   # action agent0
                0,   # action agent0
            ]),
            # agent2
            np.array([
                47,  # apples picked
                0,   # action agent0
                3,   # action agent0
                4,   # action agent0
            ]),
        )
    """

    def __init__(
        self,
        n_agents=1,
        tag_cost=0.0,
        tagged_length=10,
        max_apples=100,
        growth_rate=0.1,
    ):
        """Initialize SimpleHarvest

        Arguments:
            n_agents (int): number of agents in environment
            tag_cost (float): cost of tagging other
            tagged_length (int): time-out length when tagged
            max_apples (int): carrying capacity of apples
            growth_rate (float): growth rate in logistic growth
        """
        super().__init__()

        # Time
        self.t = 0
        self.t_limit = 3000

        # Actions
        self.n_agents = n_agents
        self.action_meanings = dict([
            *[
                (0, "NOOP"),
                (1, "PICK"),
            ],
            *[  # 0 is the agent that does the action
                (i + 2, f"TAG-{i}")
                for i in range(self.n_agents)
            ],
        ])

        self.max_apples = max_apples
        self.observation_space = self.generate_observation_space(
            self.max_apples,
            self.n_agents,
        )
        self.action_space = spaces.Discrete(self.n_agents + 2)

        # Initial observation variables
        self.initial_apples = self.max_apples // 2
        self.available_apples = self.initial_apples
        self.previous_actions = np.zeros(self.n_agents)
        self.previous_rewards = np.zeros(self.n_agents)

        # Other
        self.tag_cost = tag_cost
        self.tagged_length = tagged_length
        self.tagged_cd = np.zeros(n_agents, dtype=int)
        self.growth_rate = growth_rate

    @staticmethod
    def generate_observation_space(max_apples, n_agents):
        return spaces.MultiDiscrete([
            max_apples + 1,
            *[n_agents + 2 for i in range(1, n_agents)],
        ])

    def reset(self):
        self.t = 0
        self.available_apples = self.initial_apples
        self.previous_actions = np.zeros(self.n_agents, dtype=int)
        self.previous_rewards = np.zeros(self.n_agents)
        return self.get_obs()

    def get_obs(self, agent=0):
        return np.hstack([
            self.available_apples,
            np.roll(self.previous_actions, -agent)[1:],
        ])

    def shift_actions(self, actions):
        """Shift tag actions

        Translate actions such that t_action=i means tag agenti
        """
        tags = (actions > 1)
        t_actions = (
            actions - 2 + np.arange(self.n_agents)
        ) % self.n_agents

        shifted_actions = actions.copy()
        shifted_actions[tags] = t_actions[tags] + 2
        return shifted_actions

    def calculate_rewards(self):
        actions = self.previous_actions.copy()
        rewards = np.zeros(self.n_agents)

        # If tagged from before
        already_tagged = (self.tagged_cd > 0)
        actions[already_tagged] = 0

        # If pick apples
        attempt_pick = (actions==1)
        if any(attempt_pick):
            pick_reward = min(
                1.0,  # enough apples
                self.available_apples / attempt_pick.sum(),  # share
            )
            rewards[attempt_pick] += pick_reward

        # If tags
        tags = (actions > 1)
        rewards[tags] += self.tag_cost

        # If tagged this round
        shifted_actions = self.shift_actions(actions.copy())
        # No reward if tagged
        i_tagged = np.unique(shifted_actions[tags] - 2)
        self.tagged_cd[i_tagged] -= (self.tagged_cd[i_tagged] > 0)

        # Reduce tagged count down
        self.tagged_cd[already_tagged] -= 1

        assert all(rewards <= 1)
        
        return rewards

    @staticmethod
    def logistic_growth(population, growth_rate, capacity, to_int=True):
        '''dP / dt = r * P * (1 - P / K)'''
        # Calculate population growth
        d_population = growth_rate * population * (
            1 - population / capacity
        )

        if to_int:
            # Probabilistic rounding to integer
            d_population = int(d_population + np.random.rand())

        # Update population size
        population += d_population
        return population


    def update_available_apples(self):
        '''Apples picked plus logistic growth.

        Differential equation for logistic growth
            dP / dt = r * P * (1 - P / K)
        '''
        actions = self.previous_actions.copy()
        if any(
            action not in self.action_space
            for action in actions
        ):
            raise ValueError("Actions outside action space.")

        # Picked apples
        attempt_pick = (actions==1)
        self.available_apples -= (actions==1).sum()
        self.available_apples = max(0, self.available_apples)

        # Logistic growth
        self.available_apples = self.logistic_growth(
            population=self.available_apples,
            growth_rate=self.growth_rate,
            capacity=self.max_apples,
        )

    def step(self, *actions):
        "Actions given as tuple"

        if len(actions) != self.n_agents:
            raise ValueError(
                f"{len(actions)} actions given, "
                f"but there are {self.n_agents} agents."
            )
    
        done = False
        self.t += 1
        self.previous_actions = np.array(actions)
        info = {
            f"action{i_agent}": action
            for i_agent, action
            in enumerate(actions)
        }

        # Get rewards
        rewards = self.calculate_rewards()
        self.previous_rewards = rewards
        reward = rewards[0]
        for i_agent in range(self.n_agents):
            info[f"reward{i_agent}"] = rewards[i_agent]
        
        # Update number of apples
        self.update_available_apples()
        if self.available_apples==0:
            done = True

        # Get observations
        for i_agent in range(self.n_agents):
            info[f"obs{i_agent}"] = self.get_obs(agent=i_agent)
        obs = info["obs0"]

        return obs, reward, done, info

    def render(self, mode='human'):

        # Description of state
        shifted_actions = self.shift_actions(self.previous_actions)
        rewards = self.previous_rewards.copy()
        
        l = 8
        desc = "\n".join([
            "",
            f"Apples: {self.available_apples:3d}",
            "",
            f"Agents:   " + " ".join([
                f'{f" Agent{i_agent}":>{l}}'
                for i_agent in range(self.n_agents)
            ]),
            f"Actions:  " + " ".join([
                f"{self.get_action_meaning(shifted_actions[i_agent]):>{l}s}"
                for i_agent in range(self.n_agents)
            ]),
            f"Rewards:  " + " ".join([
                f"{rewards[i_agent]:{l}g}"
                for i_agent in range(self.n_agents)
            ]),
            "",
        ])

        # Output
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(desc)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
      
    def get_action_meaning(self, action):
        return self.action_meanings[action]


if __name__=="__main__":
    n_agents = 5
    env = SimpleHarvest(n_agents=n_agents, growth_rate=0.1)
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        actions = np.random.randint(0, 2, n_agents)
        obs, reward, done, info = env.step(*actions)
        rewards = [r for k, r in info.items() if "reward" in k]
        print("Observation:", *obs)#, "Rewards", *rewards)
