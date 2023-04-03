import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from .extractors import determine_feature_extractor
from .utils import create_mlp

def vanishing_gradient_detection(grad):
    mean_grad = grad.mean()
    std_grad = grad.std()

    if std_grad < 1e-5 or mean_grad < 1e-5 or torch.any(grad < 1e-5):
        print("Vanishing gradients detected")
        print(mean_grad)
        print(std_grad)
        print(grad)
        input()


class ANFIS(nn.Module):
    def __init__(self,
            in_dim,
            out_dim,
            layers=[64, 64],
            n_rules=8,
            membership_type="Gaussian",
            normal_dis_factor=2.0,
            order=1,
            normalize_rules=True,
            debug_vanishing=True,
        ):
        super(ANFIS, self).__init__()

        # Feature extractor
        self.feature_extractor = determine_feature_extractor(in_dim)

        # Neural Network
        self.net = create_mlp(
            self.feature_extractor.n_flatten,
            out_dim,
            layers=layers,
            act_function=nn.ReLU,
        )

        # Membership functions
        if membership_type == "Gaussian":
            # Means (centers) and Standard Deviation (widths)
            self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
        elif membership_type == "Triangular":
            # Centers (centers), and left/right points (center +/- width)
            self.register_parameter("centers", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("left_widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("right_widths", nn.Parameter(torch.rand(out_dim, n_rules) * normal_dis_factor))
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(f"ANFIS with membership type <{self.membership_type}> is not supported")
            return -1

        self.order = order
        if order == 1:
            # Learning parameters
            # Jang paper
            self.register_parameter("params", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))
            self.register_parameter("biases", nn.Parameter(torch.randn(out_dim, n_rules) * normal_dis_factor))

        # Setup the membership type
        self.membership_type = membership_type
        self.n_rules = n_rules
        self.normalize_rules = normalize_rules

        # Debugging for vanishing gradients
        if debug_vanishing == True:
            layer_hook_fn = lambda module, grad_input, grad_output: vanishing_gradient_detection(grad_input[0])
            parameter_hook_fn = lambda grad: vanishing_gradient_detection(grad)

            for layer in self.net:
                layer.register_backward_hook(layer_hook_fn)

            if membership_type == "Gaussian":
                self.centers.register_hook(parameter_hook_fn)
                self.widths.register_hook(parameter_hook_fn)
            elif membership_type == "Triangular":
                self.centers.register_hook(parameter_hook_fn)
                self.left_widths.register_hook(parameter_hook_fn)
                self.right_widths.register_hook(parameter_hook_fn)

            self.params.register_hook(parameter_hook_fn)
            self.biases.register_hook(parameter_hook_fn)

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)

        # Neural Network
        # (batch_size, out_dim)
        x = self.net(x)

        # Expand for broadcasting with rules
        # (batch_size, out_dim, n_rules)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)

        # Fuzzification
        # Apply Gaussian rules
        if self.membership_type == "Gaussian":
            # (batch_size, out_dim, n_rules)
            membership = torch.exp((-(x - self.centers)**2) / (2 * self.widths**2))

        elif self.membership_type == "Triangular":
            batch_size = x.shape[0]
            # Get right/left points of the triangles
            # (batch_size, out_dim, n_rules)
            lefts = (self.centers - self.left_widths).expand(batch_size, -1, -1)
            rights = (self.centers + self.right_widths).expand(batch_size, -1, -1)
            centers = self.centers.expand(batch_size, -1, -1)

            # Clone the membership
            membership = x.clone()

            # Perform the membership function
            membership = torch.fmax(torch.fmin((membership - lefts) / (centers - lefts), (rights - membership) / (rights - centers)), torch.tensor(0.0))

        # If the rules are going to be normalized
        if self.normalize_rules:
            # Normalize the firing levels
            # (batch_size, out_dim, n_rules)
            rule_evaluation = membership / membership.sum(dim=-1, keepdim=True)
        else:
            # (batch_size, out_dim, n_rules)
            rule_evaluation = membership

        # Multiply the input by the learnable parameters and add the biases
        # (batch_size, in_dim, n_rules)
        if self.order == 1:
            defuzz = x * self.params + self.biases
        else:
            defuzz = x

        # Multiply the rules by the fuzzified input
        # (batch_size, out_dim, n_rules)
        output = rule_evaluation * defuzz

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

        if self.order == 1:
            repr_str += f"  (learnable_params): {self.params.shape, self.params.dtype}\n"
            repr_str += f"  (learnable_biases): {self.biases.shape, self.biases.dtype}\n)"

        return repr_str
