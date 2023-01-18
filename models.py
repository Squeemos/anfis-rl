import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class NatureCnn(nn.Module):
    '''NatureCNN that learns features of the image'''
    def __init__(self, obs_space):
        super(NatureCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.randn((1, *obs_space.shape)).float()
            self.n_flatten = self.cnn(sample_obs).shape[1]

    def forward(self, obs):
        return self.cnn(obs)

class FlatExtractor(nn.Module):
    '''Does nothing but pass the input on'''
    def __init__(self, obs_space):
        super(FlatExtractor, self).__init__()

        self.n_flatten = obs_space.shape[0]
        self.layer = nn.Identity()

    def forward(self, obs):
        return self.layer(obs)


class DQN(nn.Module):
    def __init__(self, obs_space, act_space, layer_size):
        super(DQN, self).__init__()

        # Feature extractor
        if len(obs_space.shape) == 1:
            self.feature_extractor = FlatExtractor(obs_space)
        elif len(obs_space.shape) == 3:
            self.feature_extractor = NatureCnn(obs_space)
        else:
            raise NotImplementedErorr("This type of environment is not supported")

        # Neural network
        self.net = nn.Sequential(
            nn.Linear(self.feature_extractor.n_flatten, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, act_space.n),
        )

    def forward(self, obs):
        return self.net(self.feature_extractor(obs))

class ANFIS(nn.Module):
    def __init__(self, obs_space, act_space, layer_size, n_rules):
        super(ANFIS, self).__init__()

        # Feature extractor
        if len(obs_space.shape) == 1:
            self.feature_extractor = FlatExtractor(obs_space)
        elif len(obs_space.shape) == 3:
            self.feature_extractor = NatureCnn(obs_space)
        else:
            raise NotImplementedErorr("This type of environment is not supported")

        # Neural Network
        self.net = self.net = nn.Sequential(
            nn.Linear(self.feature_extractor.n_flatten, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
        )

        # Fuzzification Layer / Rules
        self.centers = nn.Parameter((torch.randn(1, n_rules, layer_size) -0.5 ) * 2)
        self.widths = nn.Parameter(torch.randn(1, n_rules, layer_size) * 2)
        self.register_parameter("centers", self.centers)
        self.register_parameter("widths", self.widths)

        # Defuzzification Layer
        self.defuzzification = nn.Linear(n_rules, act_space.n)

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)

        # Neural Network
        x = self.net(x)
        x = x.unsqueeze(1).unsqueeze(1)

        # Fuzzification
        # Apply Gaussian rules
        membership = torch.exp(-((x - self.centers) / self.widths)**2)

        # Triangular
        # membership = torch.where((x - self.centers) < 0,
        #                          (x - self.centers) / -self.widths,
        #                          torch.ones_like(x) - ((x - self.centers) / self.widths)
        # )
        # membership = torch.clamp(membership, min=0, max=1)

        # Sum the values for the rules for the output from the fuzzification
        rule_evaluation = membership.sum(dim=-1)

        # Normalize the firing levels
        rule_evaluation /= rule_evaluation.sum()

        # Defuzzification
        defuzzification = self.defuzzification(rule_evaluation)

        # Squeeze to remove the extra dimension
        return defuzzification.squeeze(1)
