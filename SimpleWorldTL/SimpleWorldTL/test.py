import torch
from torch import nn
from stable_baselines3 import A2C
import time
import gymnasium as gym
import numpy as np
import BigWorldTest20
from stable_baselines3.common.env_util import make_vec_env

policy_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\BigEnv_Normal_16workers_4map_10000000timesteps_Results\nn\policy.pth"
variables_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN\pytorch_variables.pth"
model_file_path = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Past_Results\SimpleEnvTL20_16workers_4map_6000000timesteps_newsettings_Results\nn.zip"
pth_file_path = r"C:\Users\shann\Desktop\PROGRAMMiNG\Python\Results\NN\torchver.pth"
input_dim = 20
output_dim = 2

def nnKeyChanger(model_state_dict):
    new_state_dict = {}
    
    for key in model_state_dict.keys():
        if 'log_std' in key:
            continue
        new_key = key
        if 'mlp_extractor.policy_net' in key:
            new_key = key.replace('mlp_extractor.policy_net', 'actor')
        elif 'mlp_extractor.value_net' in key:
            new_key = key.replace('mlp_extractor.value_net', 'critic')
        elif 'action_net' in key:
            new_key = key.replace('action_net', 'actor.4')
        elif 'value_net' in key:
            new_key = key.replace('value_net', 'critic.4')
        new_state_dict[new_key] = model_state_dict[key]
    return new_state_dict

class SB3ToTorchNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def actorForward(self, x):
        mean = self.actor(x)
        log_std = torch.zeros_like(mean)

        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()
        return actions
        
    def valueForward(self,x):
        state_value = self.critic(x) 
        return state_value

model = SB3ToTorchNN(input_dim, output_dim)
model_state_dict = nnKeyChanger(torch.load(policy_file_path, map_location=torch.device('cpu')))
model.load_state_dict(model_state_dict)

env = BigWorldTest20.bigMapEnv(1)

while(True):
    observation, _ = env.reset()

    done = False
    while not done:
        actual_observation = torch.from_numpy(np.array(observation)).float()
        action = model.actorForward(actual_observation)
        observation, reward, done, info, truncated = env.step(action)
        time.sleep(.480)
    print(reward)
    