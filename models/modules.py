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
    def __init__(self, in_dim, out_dim, layers=[64, 64], n_rules=8, defuzz_layers=[32, 32], membership_type="Gaussian"):
        super(ANFIS, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural Network
        self.net = create_mlp(self.feature_extractor.n_flatten, n_rules, layers=layers, act_function=nn.ReLU)

        # Membership functions
        # Gaussian: Means (centers) and Standard Deviation (widths)
        # self.register_buffer("centers", (torch.randn(n_rules) -0.5 ) * 2)
        # self.register_buffer("widths", (torch.randn(n_rules) * 2))
        self.register_parameter("centers", nn.Parameter((torch.randn(n_rules) -0.5 ) * 2))
        self.register_parameter("widths", nn.Parameter((torch.randn(n_rules) * 2)))

        # Setup the membership type
        self.membership_type = membership_type
        self.n_rules = n_rules

        # Defuzzification Layer
        self.defuzzification = create_mlp(n_rules, out_dim, layers=defuzz_layers)

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)
        # Intermediate step so we can multiply the inputs by the rules later
        intermediate = x
        batch_size = x.shape[0]

        # Neural Network
        x = self.net(x)

        # Fuzzification
        # Apply Gaussian rules
        if self.membership_type == "Gaussian":
            membership = torch.exp(-((x - self.centers) / self.widths)**2)
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(f"ANFIS with membership type <{self.membership_type}> is not supported")

        # Normalize the firing levels
        rule_evaluation = membership / membership.sum(dim=-1).reshape(-1, 1).expand((batch_size, self.n_rules))

        # Multiply the rules by the input
        # Makes the input be [batch_size, in_dim, 1]
        # Makes the rules be [batch_size, 1, n_rules]
        # Lets us broadcast for element-wise multiplication
        defuzz = intermediate.unsqueeze(2) * rule_evaluation.unsqueeze(1)

        # Sum the rules
        defuzz = defuzz.sum(dim=1)

        # Defuzzification
        defuzzification = self.defuzzification(defuzz)

        return defuzzification
