import time
import random
import math
import numpy as np

import gymnasium as gym
import SimpleWorldTL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import AdamW

class SimpleEnvA3C(nn.Module):
    def __init__(self):
        super(SimpleEnvA3C, self).__init__()
        # Actor Network
        self.actor = nn.Sequential(nn.Linear(28, 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh())
        # Critic Network
        self.critic = nn.Sequential(nn.Linear(28, 128), nn.ReLU(),nn.Linear(128, 1))
        
    def forward(self, x):
