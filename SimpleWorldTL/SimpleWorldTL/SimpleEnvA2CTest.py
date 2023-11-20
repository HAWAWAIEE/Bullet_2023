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

log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\Log"
tensorboard_log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\Tensor"
save_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\NN\nn"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
  save_freq=10000000,
  save_path=log_dir
)


def make_env(rank, seed=0):
    def _init():
        mapNum = rank % 4
        env = SimpleWorldTL28.simpleMapEnv(mapNum=mapNum)
        env = Monitor(env)
        return env
    return _init


def train():
    env_id = 'simpleMapEnv'
    num_envs = 16
    env_fns = [make_env(i) for i in range(num_envs)]

    env = SubprocVecEnv(env_fns)


    # env = make_vec_env(env_id, n_envs=16, env_kwargs=None, make_env = make_env, monitor_dir = log_dir,wrapper_class = Monitor)
    model = A2C('MlpPolicy', env, verbose=1, n_steps = 10, ent_coef=0.001,  tensorboard_log= tensorboard_log_dir)

    model.learn(total_timesteps=10000000, tb_log_name="SimpleEnv_", callback= checkpoint_callback, progress_bar=True)
    model.save(path = save_dir,include="SimpleWorldTL_")
    env.close()

    print("model finished!")
    
if __name__ == '__main__':
    train()