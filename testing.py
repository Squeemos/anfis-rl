import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models import DQN, ANFIS
from memory import Memory
from utils import wrap_input, epsilon_greedy, make_env, get_n_params
from config import Config

def main() -> int:
    conf = Config("config.yaml")
    env = make_env(conf.training.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if False:
        model = DQN(env.observation_space.shape, env.action_space.n, conf.dqn.layers).to(device)
        target = DQN(env.observation_space.shape, env.action_space.n, conf.dqn.layers).to(device)
    else:
        model = ANFIS(env.observation_space.shape, env.action_space.n, conf.anfis.layers, conf.anfis.n_rules, conf.anfis.defuzz_layers).to(device)
        target = ANFIS(env.observation_space.shape, env.action_space.n, conf.anfis.layers, conf.anfis.n_rules, conf.anfis.defuzz_layers).to(device)

if __name__ == "__main__":
    raise SystemExit(main())
