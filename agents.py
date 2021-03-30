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

    def __init__(self, observation_space, action_embedding_dim=2):
        nvec = observation_space.nvec
        self.max_apples = nvec[0]
        self.n_agents = nvec[1]
        self.n_actions = self.n_agents + 2

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


class MLAgent(nn.Module, AppleAgent):
    '''PyTorch based agent.'''

    def __init__(self, observation_space, action_embedding_dim=2):
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
