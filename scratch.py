import gym
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models import DQN, ANFIS
from memory import Memory
from utils import wrap_input, epsilon_greedy

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANFIS((1,), 1, 16, 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.MSELoss()

    for it in range(10_000):
        x = (0.5 - torch.rand((1_000, 1), device=device)) * 4
        y = x * x
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    x = torch.linspace(-2, 2, 10_000).to(device).reshape(-1, 1)
    r = x * x

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
