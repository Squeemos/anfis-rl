import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class DQN(nn.Module):
    def __init__(self, obs_space, act_space, layer_size):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_space.shape[0], layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, act_space.n),
        )

    def forward(self, obs):
        return self.net(obs)

class ANFIS(nn.Module):
    def __init__(self, obs_space, act_space, layer_size, n_rules):
        super(ANFIS, self).__init__()

        # Neural Network
        self.net = self.net = nn.Sequential(
            nn.Linear(obs_space.shape[0], layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
        )

        # Fuzzification Layer
        self.centers = nn.Parameter((torch.randn(1, n_rules, layer_size) -0.5 ) * 2)
        self.widths = nn.Parameter((torch.randn(1, n_rules, layer_size) - 0.5) * 2)
        self.register_parameter("centers", self.centers)
        self.register_parameter("widths", self.widths)

        # Rule Layer
        self.rule_layer = nn.Linear(n_rules, act_space.n)

        # Defuzzification Layer
        self.defuzzification = nn.Linear(act_space.n, act_space.n)

    def forward(self, x):
        # Neural Network
        x = self.net(x)

        # Fuzzification
        x = x.unsqueeze(1).unsqueeze(1)
        membership = torch.exp(-((x - self.centers) / self.widths)**2)
        # print("membership", membership.shape)
        intermediate = membership.sum(dim=-1)
        # print("int", intermediate.shape)
        # membership = membership.mean(dim=-1)
        # print("membership-mean", membership.shape)

        # Rule evaluation
        rule_evaluation = self.rule_layer(intermediate)
        # rule_evaluation = F.softmax(rule_evaluation, dim=0)
        # print(rule_evaluation.shape)

        # Defuzzification
        defuzzification = self.defuzzification(rule_evaluation)

        # Normalization
        # output = F.softmax(defuzzification, dim=-1).squeeze(1)
        # print("softmax", output.shape)


        return defuzzification.squeeze(1)
