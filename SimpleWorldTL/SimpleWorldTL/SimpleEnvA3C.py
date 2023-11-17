from itertools import accumulate
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

import multiprocessing

import SimpleWorldTL

# Env Settings
STATENUM = 28
ACTIONNUM = 2

# Hyper Parameters
UPDATESTEP = 100
MAXEPISODE = 99999999
MAXSTEP = 5000
GAMMA = 0.99
LEARNINGRATE = 0.001

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        # Stochastic NN for actions
        self.fcMu = nn.Linear(128, ACTIONNUM)
        self.fcSigma = nn.Linear(128, ACTIONNUM)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2*F.tanh(self.fcMu1(x))                   # action range -2~2
        sigma = F.softplus(self.fcSigma1(x)) + 0.001   # to avoid 0
        return mu, sigma
    
    # Function to sample actions from Distribution
    def selectAction(self, state):
        stateTensor = torch.tensor(state, dtype=torch.float32)
        mu, sigma = self.forward(stateTensor)
        mu1, mu2 = mu.split(ACTIONNUM, dim=1)
        sigma1, sigma2 = sigma.split(ACTIONNUM, dim=1)
        distribution1 = torch.distributions.Normal(mu1, sigma1)
        distribution2 = torch.distributions.Normal(mu2, sigma2)
        action1 = torch.clamp(distribution1.sample(), -2, 2)
        action2 = torch.clamp(distribution2.sample(), -2, 2)
        return [action1, action2], [distribution1, distribution2]
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   
    
class SimpleEnvGlobalNetwork(nn.Module):
    def __init__(self, learningRate = LEARNINGRATE):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learningRate)
        
class SimpleEnvWorker:
    def __init__(self, GlobalNetwork, env):
        # Set Worker Network Structure
        self.network = SimpleEnvGlobalNetwork()
        self.env = env
        self.globalNetwork = GlobalNetwork
        self.episodeNum = 0
        
    def download(self):
        # Update WorkerNetwork with GlobalNetwork
        self.network.load_state_dict(self.globalNetwork.state_dict())
        pass
    
    def upload(self, ActorAccumulatedGradient, CriticAccumulatedGradient):
        # Update GlobalNetwork with Accumulated Gradients

        self.globalNetwork.optimizer.zero_grad()

        # Actor와 Critic의 그라디언트를 수동으로 적용
        for globalParam, grad in zip(self.globalNetwork.actor.parameters(), ActorAccumulatedGradient):
            globalParam.grad = grad.clone()

        for globalParam, grad in zip(self.globalNetwork.critic.parameters(), CriticAccumulatedGradient):
            globalParam.grad = grad.clone()

        # 글로벌 네트워크의 가중치 업데이트
        self.globalNetwork.optimizer.step()
        pass
    
    def calculateGradient(self, opt, NStepReturn:float, state, action, actionDist):
        
        # Critic Loss
        value = self.network.critic.forward(state)
        criticLoss = F.mse_loss(value, NStepReturn)

        # Actor Loss
        dist1 = actionDist[0]
        dist2 = actionDist[1]
        action1, action2 = action
        logProb1 = dist1.log_prob(action1).sum(-1).unsqueeze(-1)
        logProb2 = dist2.log_prob(action2).sum(-1).unsqueeze(-1)
        advantage = (NStepReturn - value).detach()  # NStepReturn - V
        actorLoss = -(logProb1 * advantage + logProb2 * advantage)
        # entropy needed

        # Total Loss
        total_loss = actorLoss + criticLoss

        # Calculate Gradients for Back Propagation
        self.network.zero_grad()
        total_loss.backward()
        
        actorGradients = [param.grad.data for param in self.network.actor.parameters()]
        criticGradients = [param.grad.data for param in self.network.critic.parameters()]

        return actorGradients, criticGradients

    def work(self):
        totalstep = 0
        rewardList = []
        stateList = []
        actionList = []
        actionDistList = []
        while self.episodeNum < MAXEPISODE:
            # Episode Start
            self.env.reset()
            state = self.env.initialState
            
            # Run env
            for i in range(MAXSTEP): 
                stateList.append(state)
                stateOld = state
                action, actionDist = self.network.Actor.selectAction(stateOld)
                action = [action[0].item(), action[1].item()]
                state, reward, done = self.env.step(action)
                actionList.append(action)
                actionDistList.append(actionDist)
                rewardList.append(reward)

            # End episode if reached MAXSTEP
                if i == MAXSTEP-1:
                    done = True
                
                if totalstep%UPDATESTEP == 0 or done:
                    actorAccumulatedGradient = None
                    criticAccumulatedGradient = None
                    v = self.globalNetwork.critic.forward(state)
                    nStepReturn = v
                    
            # Update Accumulated Gradient
                    for j in range(len(stateList)-1,-1,-1):
                        nStepReturn = GAMMA*nStepReturn + rewardList[j]             
                        actorGradient, criticGradient = self.calculateGradient(nStepReturn, stateList[j], actionList[j], actionDistList[j])
                        actorAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(actorAccumulatedGradient, actorGradient)]
                        criticAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(criticAccumulatedGradient, criticGradient)]
                            
            # Push and Pull
                    self.upload(actorAccumulatedGradient, criticAccumulatedGradient)
                    self.download()
            # Initialize Trajectory Segment
                    rewardList.clear()
                    stateList.clear()
                    actionList.clear()
                    actionDistList.clear()
            # when episode ends during MAXSTEP
                    if done:
                        break
                    
                totalstep+=1  
            self.episodeNum+=1
            # Save Results 
            # Env has Episodic Reward List and Episodic Time Step List
           
        pass
        