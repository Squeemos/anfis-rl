import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class DQN(nn.Module):
    def __init__(self, obs_space, act_space):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_space.shape[0], 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, act_space.n),
        )

    def forward(self, obs):
        return self.net(obs)
