import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

import gym
import gymnasium

def wrap_input(arr, device, dtype=torch.float, reshape=False):
    output = torch.from_numpy(np.array(arr)).type(dtype).to(device)
    if reshape:
        output = output.reshape(-1, 1)

    return output

def soft_update(target, online, tau):
    with torch.no_grad():
        for target_param, local_param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return pytorch_total_params

def epsilon_greedy(start, end, n_steps, it):
    return max(start - (start - end) * (it / n_steps), end)

def make_env(env_id):
    env = gymnasium.make(env_id)

    if "ALE" in env_id:
        env = WarpFrame(env)

    return env

def create_mlp(in_dim, out_dim, layers=[], act_function=None, batch_norm=None):
    if len(layers) == 0:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    modules = [nn.Linear(in_dim, layers[0])]
    if batch_norm is not None:
        modules.append(batch_norm(layers[0]))

    if act_function is not None:
        modules.append(act_function())

    for idx in range(0, len(layers) - 1):
        modules.append(nn.Linear(layers[idx], layers[idx + 1]))
        if batch_norm is not None:
            modules.append(batch_norm(layers[idx + 1]))
        if act_function is not None:
            modules.append(act_function())

    modules.append(nn.Linear(layers[-1], out_dim))

    return nn.Sequential(*modules)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super(WarpFrame, self).__init__(env)

        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(1, self.height, self.width), dtype=env.observation_space.dtype
        )

    # Setup so that PyTorch can accept it
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame / 255.0
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[None, :, :]

# TODO: Vectorized Environments
