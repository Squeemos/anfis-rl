import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.modules import ANFIS, DQN
from models.utils import get_n_params

def main() -> int:
    batch_size = 12
    n_rules = 100
    out_dim = 5
    in_dim = 7
    inp = torch.randn(batch_size, in_dim) * 2

    layer = nn.Linear(in_dim, out_dim)

    outp = layer(inp)
    print("in", inp.shape)
    print("net out", outp.shape)

    centers = torch.randn(out_dim, n_rules)
    widths = torch.randn(out_dim, n_rules)
    params = torch.randn(out_dim, in_dim)
    biases = torch.randn(out_dim, in_dim)

    outputs = outp.unsqueeze(-1).expand(-1, -1, n_rules)

    # sqrt(2pi)
    test = (1 / (widths * 2.50662827463)) * torch.exp(-((outputs - centers) / widths)**2 / 2)
    print("expanded out", outputs.shape)
    print("rule eval", test.shape)

    rule_eval = test / test.sum(dim=-1, keepdim=True)
    print("rule normed", rule_eval.shape)

    i = inp.unsqueeze(-1).expand(-1, -1, n_rules).permute(0, 2, 1)
    print("augmented input", i.shape)

    defuzz = (rule_eval @ i)
    print("input * rules", defuzz.shape)

    j = params * defuzz + biases
    print("params * rules", j.shape)

    out = j.sum(dim=-1)
    print("final output", out.shape)

if __name__ == "__main__":
    raise SystemExit(main())
