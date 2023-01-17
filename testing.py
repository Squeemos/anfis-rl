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

    print(obs.max(), obs.min())


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
