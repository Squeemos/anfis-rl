import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.anfis import ANFIS

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def weierstrass(x, a, b, n_scale):
    if "n" not in dir(weierstrass):
        n = torch.arange(n_scale).to(x.device)
        weierstrass.an = (a**n).unsqueeze(0)
        weierstrass.bn = (b**n).unsqueeze(0)
    return torch.sum(weierstrass.an * torch.cos(weierstrass.bn * torch.pi * x), dim=1)

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = 1
    x = torch.linspace(-domain, domain, 10_000).to(device).reshape(-1, 1)
    y = weierstrass(x, .5, 3, 100)


    fig = plt.figure(figsize=(10,10))
    x = x.clone().detach().cpu().numpy()
    y = y.clone().detach().cpu().numpy()
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    raise SystemExit(main())
