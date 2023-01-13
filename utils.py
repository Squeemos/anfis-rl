import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np

def wrap_input(arr, device, dtype=torch.float):
    return torch.from_numpy(np.array(arr)).type(dtype).to(device)
