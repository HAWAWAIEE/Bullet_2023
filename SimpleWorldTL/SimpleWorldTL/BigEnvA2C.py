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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\Log"
tensorboard_log_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results"
save_dir = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\nn"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
  save_freq=10000000,
  save_path=log_dir
)

# class CustomPolicy(ActorCriticPolicy):
#     def _build_mlp_extractor(self) -> None:
#         self.mlp_extractor = CustomMLPExtractor(self.features_dim)

# class CustomMLPExtractor(nn.Module):
#     def __init__(self, features_dim: int):
#         super(CustomMLPExtractor, self).__init__()

#         self.shared_net = nn.Sequential(
#             nn.Linear(features_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#         )

    # def forward(self, features: torch.Tensor) -> torch.Tensor:
    #     return self.shared_net(features)


def make_env(rank, seed=0):
    def _init():
        mapNum = 1
        env = BigWorldTest20.bigMapEnv(mapNum=mapNum)
        env = Monitor(env)
        return env
    return _init

def train():
    env_id = 'BigMapEnvDPBA'
    num_envs = 16
    env_fns = [make_env(i) for i in range(num_envs)]

    env = SubprocVecEnv(env_fns)

    # env = make_vec_env(env_id, n_envs=16, env_kwargs=None, make_env = make_env, monitor_dir = log_dir,wrapper_class = Monitor)
    model = A2C('MlpPolicy', env, verbose=1, n_steps = 20, ent_coef=0.001,  tensorboard_log= tensorboard_log_dir)


    model.learn(total_timesteps=10000000, tb_log_name="BigEnv_", callback= checkpoint_callback, progress_bar=True)
    model.save(path = save_dir,include="SimpleWorldTL_")
    env.close()

    print("model finished!")
    
if __name__ == '__main__':
    train()