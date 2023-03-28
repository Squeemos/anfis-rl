import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

def determine_feature_extractor(in_dim):
    if len(in_dim) == 1:
        return FlatExtractor(in_dim)
    elif len(in_dim) == 3:
        return NatureCnn(in_dim)
    else:
        raise NotImplementedError("This type of input is not supported")

class NatureCnn(nn.Module):
    '''NatureCNN that learns features of the image'''
    def __init__(self, in_dim):
        super(NatureCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.randn((1, *in_dim)).float()
            self.n_flatten = self.cnn(sample_obs).shape[1]

    def forward(self, obs):
        return self.cnn(obs)

class FlatExtractor(nn.Module):
    '''Does nothing but pass the input on'''
    def __init__(self, in_dim):
        super(FlatExtractor, self).__init__()

        self.n_flatten = np.prod(in_dim)

    def forward(self, obs):
        # Reshape to be batch_size x flattened_dim
        return obs.reshape(obs.shape[0], self.n_flatten)
