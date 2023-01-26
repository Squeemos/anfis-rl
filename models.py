import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

def determine_feature_extractor(in_dim):
    if len(in_dim) == 1:
        return FlatExtractor(in_dim)
    elif len(in_dim) == 3:
        return NatureCnn(in_dim)
    else:
        raise NotImplementedErorr("This type of input is not supported")

def create_mlp(in_dim, out_dim, layers=[], act_function=None, batch_norm=None):
    if len(layers) == 0:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    modules = [nn.Linear(in_dim, layers[0])]
    if batch_norm is not None:
        modules.append(batch_norm(layers[0]))

    if act_function is not None:
        modules.append(act_function())

    for idx in range(0, len(layers) - 1):
        modules.append(nn.Linear(layers[idx], layers[idx + 1]))
        if batch_norm is not None:
            modules.append(batch_norm(layers[idx + 1]))
        if act_function is not None:
            modules.append(act_function())

    modules.append(nn.Linear(layers[-1], out_dim))

    return nn.Sequential(*modules)

class NatureCnn(nn.Module):
    '''NatureCNN that learns features of the image'''
    def __init__(self, in_dim):
        super(NatureCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.randn((1, *in_dim)).float()
            self.n_flatten = self.cnn(sample_obs).shape[1]

    def forward(self, obs):
        return self.cnn(obs)

class FlatExtractor(nn.Module):
    '''Does nothing but pass the input on'''
    def __init__(self, in_dim):
        super(FlatExtractor, self).__init__()

        self.n_flatten = in_dim[0]

    def forward(self, obs):
        return obs

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
        self.net = create_mlp(self.feature_extractor.n_flatten, n_rules, layers=layers, act_function=nn.LeakyReLU)

        # Membership functions
        # Gaussian: Means (centers) and Standard Deviation (widths)
        self.register_buffer("centers", (torch.randn(n_rules) -0.5 ) * 2)
        self.register_buffer("widths", (torch.randn(n_rules) * 2))

        # Setup the membership type
        self.membership_type = membership_type
        self.n_rules = n_rules

        # Defuzzification Layer
        self.defuzzification = create_mlp(n_rules, out_dim, layers=defuzz_layers, act_function=nn.LeakyReLU)

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
