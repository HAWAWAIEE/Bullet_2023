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
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = make_vec_env(SimpleWorldTL.simpleMapEnv, n_envs=16, env_kwargs={'mapNum': 3})
model = A2C('MlpPolicy', env, verbose=1)

save_path = "C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\Bullet_2023\\Data\\NN"
os.makedirs(save_path, exist_ok=True)

checkpoint_callback = CheckpointCallback(save_freq=10000000, save_path=save_path,
                                         name_prefix="critic_network")

model.learn(total_timesteps=1000000000, callback=checkpoint_callback)

print("model finished!")
