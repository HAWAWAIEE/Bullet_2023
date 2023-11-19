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
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(self.save_path + str(self.n_calls))
        return True

env = make_vec_env(SimpleWorldTL28.simpleMapEnv, n_envs=16, env_kwargs={'mapNum': 3})
model = A2C('MlpPolicy', env, verbose=1)

save_callback = SaveOnBestTrainingRewardCallback(check_freq=100000, save_path="C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\NN")

model.learn(total_timesteps=1000000, callback=save_callback)

print("model finished!")
