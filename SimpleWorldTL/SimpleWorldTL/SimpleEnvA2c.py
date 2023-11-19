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
import SimpleWorldTL28
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

env = make_vec_env(SimpleWorldTL28.simpleMapEnv, n_envs=16, env_kwargs={'mapNum': 3})
model = PPO('MlpPolicy', env, verbose=1)


model.learn(total_timesteps=100000)
model.save("C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\NN\\Model")
env.close()

print("model finished!")