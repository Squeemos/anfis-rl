import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from .extractors import determine_feature_extractor
from .utils import create_mlp

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=[64, 64]):
        super(DQN, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural network
        self.net = create_mlp(self.feature_extractor.n_flatten, out_dim, layers, act_function=nn.ReLU)

    def forward(self, x):
        x = self.net(self.feature_extractor(x))
        return x

class ANFIS(nn.Module):
    def __init__(self, in_dim, out_dim, layers=[64, 64], n_rules=8, membership_type="Gaussian"):
        super(ANFIS, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural Network
        self.net = create_mlp(self.feature_extractor.n_flatten, out_dim, layers=layers, act_function=nn.ReLU)

        # Membership functions
        # Gaussian: Means (centers) and Standard Deviation (widths)
        self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * 2))
        self.register_parameter("widths", nn.Parameter(torch.rand(out_dim, n_rules) * 2))

        # Learning parameters
        self.register_parameter("params", nn.Parameter(torch.randn(out_dim, n_rules) * 2))
        self.register_parameter("biases", nn.Parameter(torch.randn(out_dim, n_rules) * 2))

        # Setup the membership type
        self.membership_type = membership_type
        self.n_rules = n_rules

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)

        # Neural Network
        # (batch_size, n_rules)
        x = self.net(x)

        # Expand for broadcasting with rules
        # (batch_size, in_dim, n_rules)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)

        # Fuzzification
        # Apply Gaussian rules
        if self.membership_type == "Gaussian":
            # (batch_size, out_dim, n_rules)
            gauss = torch.exp(-((x - self.centers) / self.widths)**2 / 2)
            membership = torch.exp(gauss)
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(f"ANFIS with membership type <{self.membership_type}> is not supported")

        # Normalize the firing levels
        # (batch_size, out_dim, n_rules)
        rule_evaluation = membership / membership.sum(dim=-1, keepdim=True)

        # Multiply input by the fuzzy rules
        # (batch_size, out_dim, n_rules)
        defuzz = rule_evaluation * x

        # Learning parameters for defuzzification
        # (batch_size, out_dim, n_rules)
        output = defuzz * self.params + self.biases

        # Sum the rules
        # (batch_size, out_dim)
        return output.sum(dim=-1)
