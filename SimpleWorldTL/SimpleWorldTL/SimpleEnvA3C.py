import time
import random
import math
import numpy as np

import gymnasium as gym
import SimpleWorldTL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

import SimpleWorldTL

STATENUM = 28
ACTIONNUM = 2
UPDATESTEP = 100
WORKMAXSTEP = 1000
MAXEPISODE = 99999999
MAXSTEP = 5000

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        # Stochastic NN for action1
        self.fcMu1 = nn.Linear(128, ACTIONNUM)
        self.fcSigma1 = nn.Linear(128, ACTIONNUM)
        # Stochastic NN for action2
        self.fcMu2 = nn.Linear(128, ACTIONNUM)
        self.fcSigma2 = nn.Linear(128, ACTIONNUM)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu1 = 2*F.tanh(self.fcMu1(x))                   # action range -2~2
        sigma1 = F.softplus(self.fcSigma1(x)) + 0.001   # to avoid 0
        mu2 = 2*F.tanh(self.fcMu2(x))                   # action range -2~2
        sigma2 = F.softplus(self.fcSigma2(x)) + 0.001   # to avoid 0
        return mu1, sigma1, mu2, sigma2
    
    # Function to sample actions from Distribution
    def selectAction(self, State):
        mu1, sigma1, mu2, sigma2 = self.forward(State)
        distribution1 = torch.distribution.Normal(mu1, sigma1)
        distribution2 = torch.distribution.Normal(mu2, sigma2)
        action1 = torch.clamp(distribution1.sample(),-2,2)
        action2 = torch.clamp(distribution2.sample(),-2,2)
        return action1, action2
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   
    
class SimpleEnvGlobalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor(STATENUM, ACTIONNUM)
        self.critic = Critic(STATENUM)
        
class SimpleEnvWorker:
    def __init__(self, GlobalNetwork, env):
        # Set Worker Network Structure
        self.network = SimpleEnvGlobalNetwork(STATENUM, ACTIONNUM)
        self.env = env
        self.globalNetwork = GlobalNetwork
        self.episodeNum = 0
        
    def download(self):
        # Update WorkerNetwork with GlobalNetwork
        pass
    
    def upload(self):
        # Update GlobalNetwork with Accumulated Gradients
        pass
    
    def update(self):
        # n-step Advance Actor Critic Update and accumulate gradient
        pass

    def work(self):
        totalstep = 0
        while self.episodeNum < MAXEPISODE:
            # Episode Start
            self.env.reset()
            observation = self.env.initialState
            # Run env
            for i in range(MAXSTEP): 
                observationOld = observation
                observation, reward, done = self.env.step(self.network.Actor.selectAction(observationOld))
                
            # End episode if reached MAXSTEP
                if i == MAXSTEP-1:
                    done = True
                
            # Update WorkerNetwork
                if i%UPDATESTEP == 0 or done:
                    self.update()
            # when episode ends during MAXSTEP
                    if done:
                        break
            # Push and Pull 
                if totalstep%WORKMAXSTEP == 0:
                    self.upload()
                    self.download()
                totalstep+=1  
            self.episodeNum+=1
            # Save Results    
        pass
        