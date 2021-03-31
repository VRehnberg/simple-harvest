import numpy as np
import torch
from torch import nn

class ObservationEmbedding(nn.Module):

    def __init__(self, observation_space, action_embedding_dim):
        super().__init__()
        
        nvec = observation_space.nvec
        self.max_apples = nvec[0]
        self.n_agents = nvec[1]
        self.n_actions = self.n_agents + 2

        self.action_embedding = nn.Embedding(
            self.n_actions,
            action_embedding_dim,
        )

    def forward(x):
        # Scale number of apples to between -1 and 1
        x[:, 0] = 2.0 * x[:, 0] / self.max_apples - 1.0
        
        # Encode actions
        x = torch.hstack([
            x[:, 0],
            *[
                self.action_embedding(x[:, i_action + 1])
                for i_action in range(self.n_actions)
            ]
        ])

        return x

class AppleAgent():

    def __init__(self, observation_space):
        nvec = observation_space.nvec
        self.max_apples = nvec[0]
        self.n_actions = nvec[1]
        self.n_agents = self.n_actions - 2

    def reset(self, observation):
        self.obs = observation

    def act(self):
        a = torch.randint(self.n_actions, [])
        return a

    def update(self, action, observation, reward, done):
        self.previous_action = action
        self.previous_reward = reward
        self.obs = observation

    def new_agent(self):
        pass

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            ")"
        )

class Punisher(AppleAgent):
    '''Punishes those that take apples when below half capacity.'''

    def act(self):
        seen_apples = self.obs[0]
        seen_actions = np.asarray(self.obs[1:])
        threshold = (self.max_apples // 2)
        if seen_apples > threshold:
            # Pick apple
            action = 1
        else:
            took_apples = (seen_actions == 1)
            taken_apples = took_apples.sum()
            if (seen_apples + taken_apples) > threshold:
                # Wait
                action = 0
            elif taken_apples == 0:
                action = 0
            else:
                # Punish one of the transgressors
                print(took_apples)
                took_idx = np.flatnonzero(took_apples)
                action = np.random.choice(took_idx)

        return action

class QLearner(AppleAgent):

    def __init__(
        self,
        observation_space,
        learning_rate=1.0,
        discount=1.0,
        epsilon=0.0,
    ):
        super().__init__(observation_space)

        #TODO
        raise NotImplementedError

        # Dimensionality
        self.observation_dims = np.array(observation_dims)
        self.n_states = np.prod(observation_dims)
        self.n_actions = n_actions
        self.q_values = np.zeros([self.n_states, self.n_actions])

        # Learning parameters
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.discount = discount
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.episodic = episodic
        self.memory = memory
        self.history = np.zeros([self.memory, 3], dtype=int)

    def observe(self, observation):
        '''Update state from observation.'''
        observation = np.atleast_1d(observation)
        state = np.ravel_multi_index(
            observation,
            self.observation_dims,
        )
        self.state = state

    def act(self):
        if self.random_state.rand() > self.epsilon:
            action = np.argmax(self.q_values[self.state, :])
        else:
            action = self.random_state.choice(self.n_actions)
        return action

    def update(self, action, observation, reward, done):
        # Update states
        previous_state = self.state
        self.observe(observation)
        if not self.episodic:
            # If not episodic episode is the same as taken steps
            self.episode += 1
            self.epsilon = self.initial_epsilon / self.episode

        # If case for speed up in Chain case
        if self.memory > 1:
            # Update history
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1, :] = (previous_state, action, reward)

            # Get indices from history
            states = self.history[:self.episode, 0]
            actions = self.history[:self.episode, 1]
            rewards = self.history[:self.episode, 2]
            next_states = np.roll(states, -1)
            next_states[-1] = self.state
        else:
            self.history[-1, :] = (previous_state, action, reward)
            states = self.history[:, 0]
            actions = self.history[:, 1]
            rewards = self.history[:, 2]
            next_states = np.array([self.state])

        # Perform Q-learning step
        future_rewards = self.q_values[next_states, :].max(axis=1)
        if done:
            future_rewards[-1] = 0.0

        self.q_values[states, actions] += self.learning_rate * (
            rewards
            + self.discount * future_rewards
            - self.q_values[states, actions]
        )
        
    def reset(self, observation):
        '''Run when starting new game.'''
        if self.episodic:
            self.episode += 1
            self.epsilon = self.initial_epsilon / self.episode
        self.observe(observation)

    def new_agent(self):
        self.episode = 0
        self.learning_rate = self.initial_learning_rate
        self.epsilon = self.initial_epsilon
        self.q_values = np.zeros([self.n_states, self.n_actions])

    def __repr__(self):
        return (
            r"QLearner("
                r"$\alpha$="f"{self.learning_rate:.3g}"
                #r",$\gamma$="f"{self.discount:.3g}"
                r",$\varepsilon$="f"{self.initial_epsilon:.3g}"
            ")"
        )

class MLAgent(nn.Module, AppleAgent):
    '''PyTorch based agent.'''

    def __init__(self, observation_space, action_embedding_dim=5):
        super().__init__()

        self.observation_embedding = ObservationEmbedding(
            observation_space,
            action_embedding_dim,
        )

    def new_agent(self):
        # Reinitialize observation embedding
        for layer in self.observation_embedding.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def forward(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

