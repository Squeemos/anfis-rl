import gym
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.modules import DQN, ANFIS
from models.utils import wrap_input, epsilon_greedy

def function(x):
    return torch.exp(x) * torch.sin(x)
    # return (torch.sin(x) * x**3) / 3
    # return x * x
    # return x ** 3

def main() -> int:
    # torch.manual_seed(127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANFIS((1,), 1, layers=[32,32], n_rules=16).to(device)
    # model = DQN((1,), 1, layers=[32, 32]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001)
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.MSELoss()

    for it in range(20_000):
        x = (0.5 - torch.rand((1_000, 1), device=device)) * 8
        y = function(x)
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1_000 == 0:
            print(loss.item())

    x = torch.linspace(-4, 4, 10_000).to(device).reshape(-1, 1)
    r = function(x)

    model_y = model(x).detach().clone().cpu().numpy()
    x = x.detach().clone().cpu().numpy()
    real_y = r.detach().clone().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, model_y)
    plt.plot(x, real_y)
    plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
