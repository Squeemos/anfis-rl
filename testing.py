import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np

import gym
import gymnasium

from config import Config
from models import ANFIS, DQN
from utils import wrap_input, make_env


def main() -> int:
    env_name = "ALE/Breakout-v5"
    env = make_env(env_name)

    obs, info = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ANFIS(env.observation_space, env.action_space, 8, 2).to(device)
    model.eval()
    sample_in = wrap_input(obs, device).unsqueeze(0)
    output = model(sample_in)
    print(output.shape)


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
