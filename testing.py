import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.modules import ANFIS

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANFIS((1,), 2, layers=[32,32], n_rules=16).to(device)

    sample_input = torch.randn((3, 1), device=device) * 8
    out = model(sample_input)
    print(out)

if __name__ == "__main__":
    raise SystemExit(main())
