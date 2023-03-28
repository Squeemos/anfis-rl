import gym
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.dqn import DQN
from models.anfis import ANFIS
from models.utils import wrap_input, epsilon_greedy, get_n_params

from graph_anfis_functions import plot_anfis_rules

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANFIS((1,), 1, layers=[64,64], n_rules=32, membership_type="Gaussian").to(device)
    # model = DQN((1,), 1, layers=[32, 32]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001)
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.MSELoss()

    if show_anfis_rules:
        plot_anfis_rules(model)

    for it in range(20_000):
        x = (0.5 - torch.rand((500, 1), device=device)) * 2 * domain
        y = function(x)
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1_000 == 0:
            print(loss.item())

    x = torch.linspace(-domain, domain, 10_000).to(device).reshape(-1, 1)
    r = function(x)

    model_y = model(x).detach().clone().cpu().numpy()
    x = x.detach().clone().cpu().numpy()
    real_y = r.detach().clone().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, model_y)
    plt.plot(x, real_y)
    plt.title(f"MSE: {loss_fn(torch.from_numpy(model_y), torch.from_numpy(real_y)).item()}")
    plt.savefig(f"./graphics/{fn_number}_{domain}_{'anfis' if anfis else 'dqn'}.png")

    print(f"\n")

    if show_anfis_rules:
        plot_anfis_rules(model)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
