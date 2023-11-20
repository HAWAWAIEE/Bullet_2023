import torch
from torch import nn

def nnKeyChanger(model_state_dict):
    """
    change Stable Baselines 3's parameters to Torch basic form
    """
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
    """
    Trained A2c Global Network Form
    """
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