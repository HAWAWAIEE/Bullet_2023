import time
import random
import math
import os
import numpy as np
import pandas as pd

import gymnasium as gym
import SimpleWorldTL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

import multiprocessing
import SimpleWorldTL
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def make_env(mapNum):
    def _init():
        env = SimpleWorldTL.simpleMapEnv(mapNum)
        return env
    return _init

env = make_vec_env(make_env(3), n_envs=8)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

model.save("C:/Users/shann/Desktop/PROGRAMMING/projects/Python/Bullet_2023/Data/NN/a2c_model")

env.close()
