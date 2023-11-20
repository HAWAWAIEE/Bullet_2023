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
import BigWorldTest20
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\Log"
tensorboard_log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\Tensor"
save_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\NN\nn"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
  save_freq=10000000,
  save_path=log_dir
)


env = make_vec_env(BigWorldTest20.simpleMapEnv, n_envs=16, env_kwargs={'mapNum': 1}, monitor_dir = log_dir,wrapper_class = Monitor)
model = A2C('MlpPolicy', env, verbose=1, n_steps = 10, ent_coef=0.001,  tensorboard_log= tensorboard_log_dir)

model.learn(total_timesteps=6000000, tb_log_name="SimpleEnv_", callback= checkpoint_callback, progress_bar=True)
model.save(save_dir,"SimpleWorldTL_")
env.close()

print("model finished!")