import time
import random
import math
import os
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import multiprocessing
import SimpleWorldTL28
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

log_dir = r"C:\Users\shann\Desktop\PROGRAMMING\projects\Python\Results\Log"
tensorboard_log_dir = "C:/Users/shann/Desktop/PROGRAMMING/projects/Python/Results/Tensor"
os.makedirs(log_dir, exist_ok=True)
env = make_vec_env(SimpleWorldTL28.simpleMapEnv, n_envs=16, env_kwargs={'mapNum': 1}, monitor_dir = log_dir,wrapper_class = Monitor)
model = A2C('MlpPolicy', env, verbose=1, n_steps = 10, ent_coef=0.001,  tensorboard_log= tensorboard_log_dir)

model.learn(total_timesteps=10000000)
model.save(r"C:\Users\shann\Desktop\PROGRAMMING\projects\Python\Results\NN\nn")
env.close()

print("model finished!")