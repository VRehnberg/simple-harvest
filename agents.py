import numpy as np
import torch
from torch import nn


class ObservationEmbedding(nn.Module):

    def __init__(self, max_apples, n_agents, action_embedding_dim):
        super().__init__()

        self.max_apples = max_apples
        self.n_actions = n_agents + 2
        self.n_agents = n_agents

        self.action_embedding = nn.Embedding(
            self.n_actions,
            action_embedding_dim,
        )

    def forward(self, x):
        # Scale number of apples to between -1 and 1
        x[:, 0] = 2.0 * (x[:, 0] / self.max_apples) - 1.0

        # Encode actions
        x = torch.hstack([
            x[:, 0],
            *[
                self.action_embedding(x[:, i_action + 1])
                for i_action in range(self.n_actions)
            ]
        ])

        return x


class AppleAgent:

    def __init__(self, max_apples: int, n_agents: int):
        self.max_apples = max_apples
        self.n_actions = n_agents + 2
        self.n_agents = n_agents

        self.obs = None
        self.episode = 0

        # Parameter used in trainable inheriting classes
        self.training = True

    def reset(self, observation):
        self.obs = observation
        self.episode = 0

    def act(self):
        a = torch.randint(self.n_actions, [])
        return a

    def update(self, action, observation, reward, done):
        self.obs = observation
        self.episode += 1

    def new_agent(self):
        pass

    def train(self, mode=True):
        if hasattr(super(), "train"):
            super().train()
        self.training = mode
        return self

    def eval(self):
        return self.train(mode=False)

    def __repr__(self):
        return (
            f"{type(self).__name__}()"
        )


class Punisher(AppleAgent):
    """Punishes those that take apples when below half capacity."""

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
                took_idx = np.flatnonzero(took_apples)
                action = np.random.choice(took_idx)

        return action


class QLearner(AppleAgent):

    def __init__(
        self,
        max_apples,
        n_agents,
        learning_rate=1.0,
        learning_rate_change=0.0,
        discount=1.0,
        epsilon=0.0,
        epsilon_change=0.0,
    ):
        super().__init__(max_apples, n_agents)

        # Dimensionality
        self.observation_dims = [
            self.max_apples + 1, *[
                self.n_actions
                for _ in range(1, self.n_agents)
            ]
        ]
        self.n_states = np.prod(self.observation_dims)
        self.q_values = np.zeros([self.n_states, self.n_actions])

        self.state = None

        # Learning parameters
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_change = learning_rate_change
        self.discount = discount
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_change = epsilon_change
        self.episode = 0

    def observe(self, observation):
        """Update state from observation."""
        observation = np.atleast_1d(observation)
        state = np.ravel_multi_index(
            observation,
            self.observation_dims,
        )
        self.state = state

    def act(self):
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.q_values[self.state, :])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def update(self, action, observation, reward, done):
        super().update(action, observation, reward, done)

        # Update states
        previous_state = self.state
        self.observe(observation)  # here self.state is updated
        self.learning_rate = self.initial_learning_rate / (
            1 + self.learning_rate_change * self.episode
        )
        self.epsilon = self.initial_epsilon / (
            1 + self.epsilon_change * self.episode
        )
        self.episode += 1

        state = previous_state
        next_state = self.state

        # Perform Q-learning step
        if done:
            future_reward = 0.0
        else:
            future_reward = self.q_values[next_state, :].max()

        self.q_values[state, action] += self.learning_rate * (
            reward
            + self.discount * future_reward
            - self.q_values[state, action]
        )

    def reset(self, observation):
        """Run when starting new game."""
        self.observe(observation)

    def new_agent(self):
        self.episode = 0
        self.learning_rate = self.initial_learning_rate
        self.epsilon = self.initial_epsilon
        self.q_values = np.zeros([self.n_states, self.n_actions])

    #def __repr__(self):
    #    return (
    #        r"QLearner("
    #        r"$\alpha$="f"{self.learning_rate:.3g}"
    #        r",$\varepsilon$="f"{self.initial_epsilon:.3g}"
    #        ")"
    #    )


class MLAgent(nn.Module, AppleAgent):
    """PyTorch based agent."""

    def __init__(self, max_apples, n_agents, action_embedding_dim=5):
        super().__init__(max_apples, n_agents)

        self.action_embedding_dim = action_embedding_dim
        self.observation_embedding = ObservationEmbedding(
            self.max_apples,
            self.n_agents,
            self.action_embedding_dim,
        )

    def new_agent(self):
        # Reinitialize observation embedding
        for layer in self.observation_embedding.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self):
        raise NotImplementedError
