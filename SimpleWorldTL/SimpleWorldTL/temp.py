import torch
from torch import nn
from stable_baselines3 import A2C
import gymnasium as gym
import SimpleWorldTL28
from stable_baselines3.common.env_util import make_vec_env

policy_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN\policy.pth"
variables_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN\pytorch_variables.pth"
model_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN.zip"

env = make_vec_env(SimpleWorldTL28.simpleMapEnv, n_envs=1, env_kwargs={'mapNum': 1})

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

state_dict = model.policy.state_dict()

for name, param in state_dict.items():
    print(f"Layer: {name}, Size: {param.size()}")

obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
print("Action:", action)