import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

def main() -> int:
    batch_size = 12
    n_rules = 100
    out_dim = 5
    in_dim = 7
    n_antecedents = 2
    inp = torch.randn(batch_size, in_dim) * 2

    layer = nn.Linear(in_dim, out_dim)

    i = inp.unsqueeze(1).expand(-1, n_antecedents, -1)
    print("in", i.shape)
    outp = layer(i)
    print("net out", outp.shape)

    centers = torch.randn(out_dim, n_rules)
    widths = torch.randn(out_dim, n_rules)
    params = torch.randn(in_dim)
    biases = torch.randn(in_dim)

    outputs = outp

    # Product t-norm
    outputs = outputs.prod(dim=1)
    print("outputs after t norm", outputs.shape)
    outputs = outputs.unsqueeze(-1).expand(-1, -1, n_rules)
    print("expanded out", outputs.shape)

    # sqrt(2pi)
    test = (1 / (widths * 2.50662827463)) * torch.exp(-((outputs - centers) / widths)**2 / 2)
    print("rule eval", test.shape)

    # Rule norm
    rule_eval = test / test.sum(dim=-1, keepdim=True)
    print("rule normed", rule_eval.shape)

    # learning params
    d = inp * params
    print("input * params", d.shape)

    # Add biases
    e = d + biases
    print("ip + biases", e.shape)

    print()
    inp = e.unsqueeze(-1).expand(-1, -1, n_rules).permute(0, 2, 1)
    print("augmented input", inp.shape)
    print("rule normed", rule_eval.shape)

    defuzz = rule_eval @ inp
    print("defuzzed", defuzz.shape)

    out = defuzz.sum(dim=-1)
    print("final output", out.shape)

if __name__ == "__main__":
    raise SystemExit(main())
