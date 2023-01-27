import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchviz import make_dot

from models.agents import Agent

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent("anfis", "CartPole-v1", [64, 64], "adam", 1e-4, "mse", 10_000, device)

if __name__ == "__main__":
    raise SystemExit(main())
