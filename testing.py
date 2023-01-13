import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np

import gym

from config import Config
from models import ANFIS, DQN

from utils import wrap_input

def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

def main() -> int:
    conf = Config("config.yaml")

    env = gym.make(conf.training.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ANFIS(env.observation_space, env.action_space, conf.anfis.layer_size, conf.anfis.n_rules).to(device)
    # model = DQN(env.observation_space, env.action_space, conf.dqn.layer_size).to(device)
    other_model = DQN(env.observation_space, env.action_space, conf.dqn.layer_size).to(device)

    model.eval()
    other_model.eval()

    print(get_n_params(model))
    print(get_n_params(other_model))


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
