import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchviz import make_dot

from models.modules import ANFIS, DQN
from models.utils import get_n_params

def main() -> int:
    batch_size = 2
    n_rules = 5
    out_dim = 2
    in_dim = 4
    inp = torch.randn(batch_size, in_dim) * 2

    layer = nn.Linear(in_dim, out_dim)
    act = nn.ReLU()

    outp = layer(inp)
    outp = act(outp)
    print("in", inp.shape)
    print("net out", outp.shape)

    centers = torch.randn(out_dim, n_rules)
    widths = torch.randn(out_dim, n_rules)
    params = torch.randn(out_dim, n_rules)

    outputs = outp.unsqueeze(-1).expand(-1, -1, n_rules)

    test = torch.exp(-((outputs - centers) / widths)**2)
    print("expanded out", outputs.shape)
    print("rule eval", test.shape)

    rule_eval = test / test.sum(dim=-1, keepdim=True)
    print("rule normed", rule_eval.shape)

    print(rule_eval.shape)



if __name__ == "__main__":
    raise SystemExit(main())
