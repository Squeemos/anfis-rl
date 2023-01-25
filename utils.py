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
