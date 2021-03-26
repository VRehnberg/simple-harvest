from collections import defaultdict
import numpy as np

from gym import Env, spaces

class SimpleHarvest(Env):
    """Simplified version of the Harvest Game in arXiv:1803.08884.

    This game is no longer a gridworld game.

    Observation:
        Apples available
        Apples picked by agents
        Punishments received
        Punishments given

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
        initially 50 apples
        memory = 2  # how far back the history is remembered
        growth_rate = 0.0  # no new apples will appear
        action history:
            0, 1, 3  # this is forgotten as memory is only 2
            4, 1, 1
            4, 0, 3
        observations given history:
            tuple(
                47, # apples remaining
                np.array([  # agent0, agent1, agent2
                    [0, 1, 1],  # apples picked
                    [0, 0, 1],  # Punishment received by agent0
                    [0, 0, 0],  # Punishment received by agent1
                    [2, 0, 0],  # Punishment received by agent2
                    [0, 0, 2],  # Punishment given by agent0
                    [0, 0, 0],  # Punishment given by agent1
                    [1, 0, 0],  # Punishment given by agent2
                ], dtype=np.uint8)
            )
                
        
    """

    def __init__(
        self,
        n_agents=2,
        memory=5,
        punish_cost=0.0,
        punished_cost=-2.0,
        max_apples=100,
        growth_rate=0.5,
    ):
        """Initialize SimpleHarvest

        Arguments:
            n_agents (int): number of agents in environment
            memory (int): how far back the agents remember
            punish_cost (float): cost of punishing other
            punished_cost (float): cost when being punished
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
        self.observation_space = (
            spaces.Discrete(self.max_apples + 1)
            spaces.Box(
                low=0,
                high=self.memory,
                shape=(1 + 2 * n_agents, n_agents),
                dtype=np.uint8,
            ),
        )

        # Initial observations
        self.available_apples = self.observation_space[0].n // 2
        self.remembered_history = np.zeros(
            *self.observation_space[1].shape
        )
        (
            self.picked_apples,
            self.have_punished,
            self.been_punished,
        ) = np.array_split(
                self.remembered_history, 
                indices_or_sections=[1, -self.n_agents],
                axis=0,
        )

        # Other
        self.punish_cost = punish_cost
        self.punished_cost = punished_cost
        self.growth_rate = growth_rate

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self, agent=0):
        agent_remembered_history = np.roll(
            self.remembered_history,
            shift=-agent,
            axis=0,
        )
        return self.available_apples, agent_remembered_history

    def logistic_growth(self)
        '''Logistic growth.

        Differential equation
            dP/dt = r*P*(1 - P / K)
        '''
        growth_factor = 1.0 + self.growth_rate
        filled_capacity = self.available_apples / self.max_apples
        self.available_apples *= int(np.round(
            growth_factor * (1.0 - filled_capacity)
        ))

    def step(self, *actions):
        "Actions given as tuple"

        if len(actions) != self.n_agents:
            raise ValueError(
                f"{len(actions)} actions given, "
                f"but there are {self.n_agents} agents."
            )

        done = False
        self.t += 1

        # Update number of apples
        actions = np.array(actions)
        attempt_pick = (actions==1)
        self.available_apples -= (actions==1).sum()
        if self.available_apples<=0:
            self.available_apples = 0
            done = True
        else:
            self.logisitic_growth()

        # Update history and get observation
        #TODO
        obs = self.getObs()

        if self.t >= self.t_limit:
          done = True

        if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
          done = True

        otherObs = None
        if self.multiagent:
          if self.from_pixels:
            otherObs = cv2.flip(obs, 1) # horizontal flip
          else:
            otherObs = self.game.agent_left.getObservation()

        info = {
          'ale.lives': self.game.agent_right.lives(),
          'ale.otherLives': self.game.agent_left.lives(),
          'otherObs': otherObs,
          'state': self.game.agent_right.getObservation(),
          'otherState': self.game.agent_left.getObservation(),
        }

        if self.survival_bonus:
          return obs, reward+0.01, done, info
        return obs, reward, done, info

    def init_game_state(self):
      self.t = 0
      self.game.reset()

    def reset(self):
      self.init_game_state()
      return self.getObs()

    def checkViewer(self):
      # for opengl viewer
      if self.viewer is None:
        checkRendering()
        self.viewer = rendering.SimpleImageViewer(maxwidth=2160) # macbook pro resolution

    def render(self, mode='human', close=False):

      if PIXEL_MODE:
        if self.canvas is not None: # already rendered
          rgb_array = self.canvas
          self.canvas = None
          if mode == 'rgb_array' or mode == 'human':
            self.checkViewer()
            larger_canvas = upsize_image(rgb_array)
            self.viewer.imshow(larger_canvas)
            if (mode=='rgb_array'):
              return larger_canvas
            else:
              return

        self.canvas = self.game.display(self.canvas)
        # scale down to original res (looks better than rendering directly to lower res)
        self.canvas = downsize_image(self.canvas)

        if mode=='state':
          return np.copy(self.canvas)

        # upsampling w/ nearest interp method gives a retro "pixel" effect look
        larger_canvas = upsize_image(self.canvas)
        self.checkViewer()
        self.viewer.imshow(larger_canvas)
        if (mode=='rgb_array'):
          return larger_canvas

      else: # pyglet renderer
        if self.viewer is None:
          checkRendering()
          self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.game.display(self.viewer)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
      if self.viewer:
        self.viewer.close()
      
    def get_action_meanings(self):
      return [self.atari_action_meaning[i] for i in self.atari_action_set]
