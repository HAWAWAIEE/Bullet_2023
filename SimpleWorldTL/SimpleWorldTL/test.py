import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import MlpExtractor
import gymnasium as gym
import numpy as np

policy_file_path = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\SimpleEnvTL28_16workers_4maps_10000000timesteps_Results\NN\policy.pth"
STATENUM = 20
observation_space = gym.spaces.Box(low=-100, high=100, shape=(20,), dtype=np.float32)
action_space = gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import get_device

class CustomActorCriticModel(nn.Module):
    def __init__(self, feature_dim, action_space, net_arch, activation_fn, device):
        super(CustomActorCriticModel, self).__init__()

        self.mlp_extractor = MlpExtractor(
            feature_dim=feature_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            device=device
        )

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def forward(self, obs):
        latent_pi, latent_vf = self.mlp_extractor(obs)

        action = self.action_net(latent_pi)

        value = self.value_net(latent_vf)

        return action, value


feature_dim = 20
net_arch = dict(pi=[64, 64], vf=[64, 64])
activation_fn = nn.ReLU
device = get_device("cpu")

model = CustomActorCriticModel(feature_dim, action_space, net_arch, activation_fn, device)

policy_state_dict = torch.load(policy_file_path, map_location=torch.device('cpu'))
model.load_state_dict(policy_state_dict)

example_input = torch.randn(1, observation_space.shape[0])
action, value = model(example_input)
print("Action probabilities:", action)
print("Value:", value)