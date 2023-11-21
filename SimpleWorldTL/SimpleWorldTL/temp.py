import torch
from torch import nn
from stable_baselines3 import A2C
import gymnasium as gym
import BigWorldTest20
from stable_baselines3.common.env_util import make_vec_env
from utils import(SB3ToTorchNN, nnKeyChanger)
import numpy as np
policy_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\Last_SimleEnvTL20\nn\policy.pth"
variables_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN\pytorch_variables.pth"
model_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN.zip"
input_dim = 20
output_dim = 2

model = SB3ToTorchNN(input_dim, output_dim)
model_state_dict = nnKeyChanger(torch.load(policy_file_path, map_location=torch.device('cpu')))
model.load_state_dict(model_state_dict)
env = BigWorldTest20.bigMapEnv(1)

while(True):
    observation = env.reset()
    done = False
    while not done:
        actual_observation = torch.from_numpy(np.array(observation)).float()
        action = model.actorForward(torch.from_numpy(observation).float())
        print(model.criticForward)
        observation, reward, done, info = env.step(action)