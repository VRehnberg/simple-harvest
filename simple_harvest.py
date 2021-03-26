import sys
import numpy as np
from io import StringIO
from contextlib import closing
from collections import defaultdict

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
        PUNISH-N (punish agent N from agents viewpoint)

    Reward:
        WAIT: 0
        PICK: +1 if apples_available else 0
        PUNISH: -punish_cost
        PUNISHED: -punished_cost

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
        n_agents=2,
        punish_cost=0.0,
        punished_cost=-2.0,
        max_apples=100,
        growth_rate=0.5,
    ):
        """Initialize SimpleHarvest

        Arguments:
            n_agents (int): number of agents in environment
            punish_cost (float): cost of punishing other
            punished_cost (float): cost when being punished
            max_apples (int): carrying capacity of apples
            growth_rate (float): growth rate in logistic growth
        """

        # Time
        self.t = 0
        self.t_limit = 3000
        self.memory = memory

        # Actions
        self.n_agents
        self.action_space = spaces.Discrete(self.n_agents + 2)
        self.action_meanings = dict(
            **{
                0 : "NOOP",
                1 : "PICK",
            },
            **{  # 0 is the agent that does the action
                i + 2 : "PUNISH-{i}"
                for i in range(self.n_agents)
            },
        )

        self.max_apples = max_apples
        self.observation_space = spaces.MultiDiscrete([
            self.max_apples + 1,
            *[self.n_agents + 2 for i in range(self.n_agents)],
        ])

        # Initial observation variables
        self.initial_apples = self.max_apples // 2
        self.available_apples = self.initial_apples
        self.previous_actions = np.zeros(self.n_agents)
        self.previous_rewards = np.zeros(self.n_agents)

        # Other
        self.punish_cost = punish_cost
        self.punished_cost = punished_cost
        self.growth_rate = growth_rate

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0
        self.available_apples = self.initial_apples
        self.previous_actions = np.zeros(self.n_agents)
        self.previous_rewards = np.zeros(self.n_agents)
        return self.get_obs()

    def get_obs(self, agent=0):
        return np.hstack(
            self.available_apples,
            np.roll(self.previous_actions, -agent),
        )
        
        return self.available_apples, agent_remembered_history

    @staticmethod
    def shift_actions(actions):
        """Shift punishing actions

        Translate actions such that p_action=i means punish agenti
        """
        punishes = (actions > 1)
        p_actions = (
            actions - 2 + np.arange(self.n_agents)
        ) % self.n_agents

        shifted_actions = actions
        shifted_actions[punishes] = p_actions[punishes]
        return shifted_actions


    def get_rewards(self):
        actions = self.previous_actions
        rewards = np.zeros(self.n_agents)

        # If pick apples
        attempt_pick = (actions==1)
        pick_reward = min(
            1.0,  # enough apples
            self.available_apples / attempt_pick.sum(),  # share
        )
        rewards[attempt_pick] += pick_reward

        # If punishes
        punishes = (actions > 1)
        rewards[punishes] += self.punish_cost

        # If punished
        shifted_actions = self.shift_actions(actions)
        # Count punishments for each agent
        i_punished, punishments = np.unique(
            shifted_actions[punishes],
            return_counts=True,
        )
        rewards[i_punished] = punishments
        
        return rewards

    def update_available_apples(self)
        '''Apples picked plus logistic growth.

        Differential equation for logistic growth
            dP / dt = r * P * (1 - P / K)
        '''
        actions = self.previous_actions

        # Picked apples
        attempt_pick = (actions==1)
        self.available_apples -= (actions==1).sum()
        self.available_apples = max(0, self.available_apples)

        # Logistic growth
        growth_factor = 1.0 + self.growth_rate
        filled_capacity = self.available_apples / self.max_apples
        self.available_apples *= (
            growth_factor * (1.0 - filled_capacity)
        )
        self.available_apples = max(
            0,
            int(np.rint(self.available_apples))
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
        info = {}

        # Get rewards
        rewards = self.get_rewards()
        self.previous_rewards = rewards
        reward = rewards[0]
        for i_agent in range(self.n_agents):
            info["reward{i_agent}"] = reward[i_agent]
        
        # Update number of apples
        self.previous_actions = actions
        self.update_available_apples()
        if self.available_apples==0:
            done = True

        # Get observations
        obs0 = self.get_obs(agent=0)
        for i_agent in range(self.n_agents):
            info["obs{i_agent}"] = self.get_obs(agent=i_agent)

        return obs, reward, done, info

    def render(self, mode='human'):

        # Description of state
        shifted_actions = self.shift_actions(self.previous_actions)
        rewards = self.previous_rewards
        desc = (
            f"Apples: {self.available_apples:3d}\n",
            "\n"
            f"Actions:  " "  ".join([
                f"Agent{i_agent}"
                for i_agent in range(self.n_agents)
            ]) "\n"
            f"          " "  ".join([
                f"{self.get_action_meaning(shifted_actions[i_agent]):>7s}"
                for i_agent in range(self.n_agents)
            ]) "\n"
            f"          " "  ".join([
                f"{rewards[i_agent]:7g}"
                for i_agent in range(self.n_agents)
            ]) "\n"
        )

        # Output
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(desc)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
      
    def get_actions_meaning(self, action):
        return self.action_meaning[action]
