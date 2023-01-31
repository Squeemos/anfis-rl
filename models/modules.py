import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from .extractors import determine_feature_extractor
from .utils import create_mlp

SQRT2PI = 2.5066282746310002

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
    def __init__(self, in_dim, out_dim, layers=[64, 64], n_rules=8, n_antecedents=1, membership_type="Gaussian"):
        super(ANFIS, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural Network
        self.net = create_mlp(self.feature_extractor.n_flatten, out_dim, layers=layers, act_function=nn.ReLU)

        # Membership functions
        # Gaussian: Means (centers) and Standard Deviation (widths)
        self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * 2))
        self.register_parameter("widths", nn.Parameter(torch.rand(out_dim, n_rules) * 2))
        # Learning parameters to multiply by the input, biases to add
        self.register_parameter("params", nn.Parameter(torch.randn(self.feature_extractor.n_flatten) * 2))
        self.register_parameter("biases", nn.Parameter(torch.randn(self.feature_extractor.n_flatten) * 2))

        # Setup the membership type
        self.membership_type = membership_type
        self.n_rules = n_rules
        self.n_antecedents = n_antecedents

    def forward(self, x):
        batch_size = x.shape[0]
        # Extract features
        x = self.feature_extractor(x)
        # Intermediate step so we can multiply the inputs by the rules later
        inputs = x

        # (batch_size, n_antecedents, out_dim)
        x = x.unsqueeze(1).expand(-1, self.n_antecedents, -1)

        # Neural Network
        # (batch_size, n_antecedents, out_dim)
        x = self.net(x)

        # Product t-norm
        # (batch_size, out_dim)
        x = x.prod(dim=1)

        # Expand for broadcasting with rules
        # (batch_size, out_dim, n_rules)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)

        # Fuzzification
        # Apply Gaussian rules
        if self.membership_type == "Gaussian":
            # (batch_size, out_dim, n_rules)
            membership = (1 / (self.widths * SQRT2PI)) * torch.exp(-((x - self.centers) / self.widths)**2 / 2)
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(f"ANFIS with membership type <{self.membership_type}> is not supported")

        # Normalize the firing levels
        # (batch_size, out_dim, n_rules)
        rule_evaluation = membership / membership.sum(dim=-1, keepdim=True)

        # Multiply input by learning params and add biases
        inp = inputs * self.params + self.biases

        # Augment input to multiply by rules, then transpose the last 2 dimensions for multiplication
        augmented = inp.unsqueeze(-1).expand(-1, -1, self.n_rules).permute(0, 2, 1)

        # Multiply the rules by the inputs
        defuzz = rule_evaluation @ augmented

        # Sum the rules
        return defuzz.sum(dim=-1)
