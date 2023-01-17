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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v1"
    env = make_env(env_name)

    obs, info = env.reset()

    state = wrap_input(obs, device).unsqueeze(0)

    model = ANFIS(env.observation_space, env.action_space, 40, 16).to(device)
    model.eval()

    output = model(state)
    print(output)
    print(output.shape)


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
