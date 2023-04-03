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

def weierstrass(x, a, b, n_scale):
    if "n" not in dir(weierstrass):
        n = torch.arange(n_scale).to(x.device)
        weierstrass.an = (a**n).unsqueeze(0)
        weierstrass.bn = (b**n).unsqueeze(0)
    return torch.sum(weierstrass.an * torch.cos(weierstrass.bn * torch.pi * x), dim=1)

def blancmange(x, n_scale):
    if "n_2" not in dir(blancmange):
        blancmange.n_2 = (2**torch.arange(n_scale)).to(x.device).unsqueeze(0)

    n_x = blancmange.n_2 * x
    frac = torch.abs(torch.round(n_x) - n_x)
    summed = torch.sum(frac / blancmange.n_2, dim=1)
    return summed

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    functions = {
        0 : lambda x: (torch.sin(x) * x**3) / 3,
        1 : lambda x: x**2,
        2 : lambda x : torch.sin(2 * torch.sin(2 * torch.sin(2 * torch.sin(x)))),
        3 : lambda x: torch.log(torch.abs(x)) * torch.sin(x),
        4 : lambda x: (((torch.sin(x) * x**3) / 3) * (torch.exp(-x) / (.01 + torch.exp(-x)))) / 2,
        5 : lambda x: weierstrass(x, .5, 3, 100).to(x.device).reshape(-1, 1),
        6 : lambda x: blancmange(x, 20).to(x.device).reshape(-1, 1),
    }

    fn_number = 3
    function = functions[fn_number]
    anfis = True
    show_anfis_rules = False
    # [-domain, domain]
    domain = 5
    training_vals = 1_000
    layer_size = 16
    n_layers = 2
    layers = list(layer_size for _ in range(n_layers))
    n_rules = 32

    # Same size models
    if anfis:
        model_1 = ANFIS((1,), 1, layers=layers, n_rules=n_rules, membership_type="Gaussian", order=1, normalize_rules=True).to(device)
        model_2 = ANFIS((1,), 1, layers=layers, n_rules=n_rules, membership_type="Gaussian", order=0, normalize_rules=True).to(device)
        model_3 = ANFIS((1,), 1, layers=layers, n_rules=n_rules, membership_type="Gaussian", order=0, normalize_rules=False).to(device)
        model_4 = ANFIS((1,), 1, layers=layers, n_rules=n_rules, membership_type="Gaussian", order=1, normalize_rules=False).to(device)
    else:
        model = DQN((1,), 1, layers=[64, 65]).to(device)

    optimizer_1 = optim.Adam(model_1.parameters(), lr=.001)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=.001)
    optimizer_3 = optim.Adam(model_3.parameters(), lr=.001)
    optimizer_4 = optim.Adam(model_4.parameters(), lr=.001)
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.MSELoss()

    models = [model_1, model_2, model_3, model_4]
    optims = [optimizer_1, optimizer_2, optimizer_3, optimizer_4]

    for it in range(20_000):
        x = (0.5 - torch.rand((training_vals, 1), device=device)) * 2 * domain
        y = function(x)

        for model, optimizer in zip(models, optims):
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 1_000 == 0:
                print(loss.item())
        if it % 1_000 == 0:
            print()

    x = torch.linspace(-domain, domain, 10_000).to(device).reshape(-1, 1)
    r = function(x)

    model_1_y = model_1(x).detach().clone().cpu().numpy()
    model_2_y = model_2(x).detach().clone().cpu().numpy()
    model_3_y = model_3(x).detach().clone().cpu().numpy()
    model_4_y = model_4(x).detach().clone().cpu().numpy()
    x = x.detach().clone().cpu().numpy()
    real_y = r.detach().clone().cpu().numpy()

    model_1_loss = loss_fn(torch.from_numpy(model_1_y), torch.from_numpy(real_y))
    model_2_loss = loss_fn(torch.from_numpy(model_2_y), torch.from_numpy(real_y))
    model_3_loss = loss_fn(torch.from_numpy(model_3_y), torch.from_numpy(real_y))
    model_4_loss = loss_fn(torch.from_numpy(model_3_y), torch.from_numpy(real_y))
    format_str = ".4f"

    fig = plt.figure(figsize=(10, 10), dpi=300.0)
    plt.plot(x, model_1_y, label="1st order with normalization")
    plt.plot(x, model_2_y, label="0th order with normalization")
    plt.plot(x, model_3_y, label="1st order without normalization")
    plt.plot(x, model_4_y, label="0th order without normalization")
    plt.plot(x, real_y, label="Expected output", linewidth=2.5)
    plt.xlabel(f"Final Loss:\n1st order with normalization: {model_1_loss.item():{format_str}}\n0th order with normalization: {model_2_loss.item():{format_str}}\n1st order without normalization: {model_3_loss.item():{format_str}}\n0th order without normalization: {model_4_loss.item():{format_str}}")
    plt.legend()
    plt.savefig(f"./graphics/{fn_number}_{domain}_{n_layers}_{n_rules}_{layer_size}_{'anfis' if anfis else 'dqn'}.png", bbox_inches='tight')

    print(f"\n")

    if show_anfis_rules:
        plot_anfis_rules(model)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
