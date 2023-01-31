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
    batch_size = 3
    n_rules = 5
    out_dim = 2
    in_dim = 4
    inp = torch.randn(batch_size, in_dim) * 2

    layer = nn.Linear(in_dim, out_dim)
    act = nn.ReLU()

    outp = layer(inp)
    outp = act(outp)
    print(inp.shape)

    centers = torch.randn(out_dim, n_rules)
    widths = torch.randn(out_dim, n_rules)
    params = torch.randn(out_dim, n_rules)

    outputs = outp.unsqueeze(-1).expand((-1, -1, n_rules))

    test = torch.exp(-((outputs - centers) / widths)**2)

    print(test.shape)

    # Normalized weights
    rules = test / test.sum(dim=-1).unsqueeze(-1).expand((-1, -1, n_rules))

    print(rules.shape)

    rp = rules * params

    print()
    print()
    x = inp.unsqueeze(-1).unsqueeze(1)
    print(x.shape)

    y = rp.unsqueeze(2)
    print(y.shape)
    print(y)

    print()
    print()
    o = x * y
    # (batch_size, out_dim, in_dim, rules)
    print("O")
    print(o.shape)
    print()

    a = o.reshape(-1, out_dim, n_rules, in_dim).prod(dim=-1).sum(dim=-1)
    b = o.sum(dim=-1).prod(dim=-1)
    c = o.reshape(-1, out_dim, n_rules, in_dim).sum(dim=-1).prod(dim=-1)
    o = o.prod(dim=-1).sum(dim=-1)

    print(a)
    print(b)
    print(c)
    print(o)


if __name__ == "__main__":
    raise SystemExit(main())
