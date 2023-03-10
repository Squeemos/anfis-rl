import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from .extractors import determine_feature_extractor
from .utils import create_mlp

class ANFIS(nn.Module):
    def __init__(self, in_dim, out_dim, layers=[64, 64], n_rules=8, membership_type="Gaussian", normal_dis_factor=2.0):
        super(ANFIS, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural Network
        self.net = create_mlp(self.feature_extractor.n_flatten, out_dim, layers=layers, act_function=nn.ReLU)

        # Membership functions
        if membership_type == "Gaussian":
            # Means (centers) and Standard Deviation (widths)
            self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
        elif membership_type == "Triangular":
            # Centers (centers), and left/right points (center +- width)
            self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("left_widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("right_widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(f"ANFIS with membership type <{self.membership_type}> is not supported")
            return -1

        # Learning parameters
        self.register_parameter("params", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
        self.register_parameter("biases", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))

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
            # We ignore the (1 / std * sqrt(2 * pi))
            gauss = (-(x - self.centers)**2 / (2 * self.widths**2))

            # (batch_size, out_dim, n_rules)
            membership = torch.exp(gauss)

        elif self.membership_type == "Triangular":
            batch_size = x.shape[0]
            # Get right/left points of the triangles
            lefts = self.centers - self.left_widths
            rights = self.centers + self.right_widths

            lefts = lefts.expand(batch_size, -1, -1)
            rights = rights.expand(batch_size, -1, -1)
            centers = self.centers.expand(batch_size, -1, -1)

            # Maybe swap to using torch.fmin instead (might save some compute time)
            # triangular_mf = torch.fmax(torch.fmin((x - a) / (b - a), (c - x) / (c - b)), torch.tensor(0.0))

            # Remove the first dimension because it's not needed
            membership = x.clone()

            # Mask out the values that aren't inside the triangles
            membership[membership < rights] = 0
            membership[membership > lefts] = 0

            # Create a mask for the values inside the specific portions of the triangle
            left_mask = (membership >= lefts) & (membership < centers)
            right_mask = (membership >= centers) & (membership <= rights)

            # Apply the mask and calculate based on the form
            # (batch_size, out_dim, n_rules)
            membership[left_mask] = (membership[left_mask] - lefts[left_mask]) / (centers[left_mask] - lefts[left_mask])
            membership[right_mask] = (rights[right_mask] - membership[right_mask]) / (rights[right_mask] - centers[right_mask])

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

    def __repr__(self):
        repr_str = super(ANFIS, self).__repr__()[:-2] + "\n"
        repr_str += f"  (n_rules): {self.n_rules}\n"

        if self.membership_type == "Gaussian":
            repr_str += f"  (centers): {self.centers.shape, self.centers.dtype}\n"
            repr_str += f"  (widths): {self.widths.shape, self.widths.dtype}\n"
        elif self.membership_type == "Triangular":
            repr_str += f"  (centers): {self.centers.shape, self.centers.dtype}\n"
            repr_str += f"  (left_widths): {self.left_widths.shape, self.left_widths.dtype}\n"
            repr_str += f"  (right_widths): {self.right_widths.shape, self.right_widths.dtype}\n"

        repr_str += f"  (learnable_params): {self.params.shape, self.params.dtype}\n"
        repr_str += f"  (learnable_biases): {self.biases.shape, self.biases.dtype}\n)"

        return repr_str
