from itertools import accumulate
import time
import random
import math
import numpy as np
import pandas as pd
import os

import gymnasium as gym
import SimpleWorldTL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

import multiprocessing
import SimpleWorldTL
import cProfile
import tkinter as tk
# Env Settings
STATENUM = 27
ACTIONNUM = 2

# Hyper Parameters
UPDATESTEP = 50
MAXEPISODE = 1000
MAXSTEP = 1000
GAMMA = 0.99
LEARNINGRATE = 0.001
ENTROPYWEIGHT = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        # Stochastic NN for actions
        self.fcMu = nn.Linear(128, ACTIONNUM)
        self.fcSigma = nn.Linear(128, ACTIONNUM)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2*F.tanh(self.fcMu(x))                   # action range -2~2
        sigma = F.softplus(self.fcSigma(x)) + 0.001   # to avoid 0
        return mu, sigma
    
    # Function to sample actions from Distribution
    def selectAction(self, state):
        stateTensor = torch.tensor(state, dtype=torch.float32)
        mu, sigma = self.forward(stateTensor)
        mu1, mu2 = mu[0], mu[1]
        sigma1, sigma2 = sigma[0], sigma[1]
        distribution1 = torch.distributions.Normal(mu1, sigma1)
        distribution2 = torch.distributions.Normal(mu2, sigma2)
        action1 = torch.clamp(distribution1.sample(), -2, 2)
        action2 = torch.clamp(distribution2.sample(), -2, 2)
        return [action1, action2]
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATENUM, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   
    
class SimpleEnvGlobalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()
        
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=LEARNINGRATE, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                
# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=LEARNINGRATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
#         super(SharedAdam, self).__init__(
#             params, lr=lr, betas=betas, eps=eps, 
#             weight_decay=weight_decay, amsgrad=amsgrad)
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['shared_step'] = torch.zeros(1).share_memory_()
#                 state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
#                 state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
#                 if weight_decay:
#                     state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
#                 if amsgrad:
#                     state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

#     def step(self, closure=None):
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 self.state[p]['steps'] = self.state[p]['shared_step'].item()
#                 self.state[p]['shared_step'] += 1
#         super().step(closure)

class SimpleEnvWorker(mp.Process):
    def __init__(self, GlobalNetwork, env, workerid, opt):
        super().__init__()
        # Set Worker Network Structure
        self.network = SimpleEnvGlobalNetwork()
        self.env = env
        self.globalNetwork = GlobalNetwork
        self.episodeNum = 0
        self.id = workerid
        self.opt = opt
        
    def download(self):
        # Update WorkerNetwork with GlobalNetwork
        self.network.load_state_dict(self.globalNetwork.state_dict())
        pass
    
    def upload(self, ActorAccumulatedGradient, CriticAccumulatedGradient):
        # Update GlobalNetwork with Accumulated Gradients

        self.opt.zero_grad()

        for globalParam, grad in zip(self.globalNetwork.actor.parameters(), ActorAccumulatedGradient):
            globalParam.grad = grad.clone()

        for globalParam, grad in zip(self.globalNetwork.critic.parameters(), CriticAccumulatedGradient):
            globalParam.grad = grad.clone()

        self.opt.step()
        pass
    
    def calculateGradient(self, NStepReturn:float, state, action):
        
        # Critic Loss
        value = self.network.critic.forward(state)
        criticLoss = F.mse_loss(value, NStepReturn)
        
        # Actor Loss
        mu, sigma = self.network.actor.forward(state)
        mu1, mu2 = mu[0], mu[1]
        sigma1, sigma2 = sigma[0], sigma[1]
        distribution1 = torch.distributions.Normal(mu1, sigma1)
        distribution2 = torch.distributions.Normal(mu2, sigma2)
        [action1, action2] = action
        action1 = torch.tensor([action1], dtype=torch.float32)
        action2 = torch.tensor([action2], dtype=torch.float32)
        logProb1 = distribution1.log_prob(action1).sum(-1).unsqueeze(-1)
        logProb2 = distribution2.log_prob(action2).sum(-1).unsqueeze(-1)
        advantage = (NStepReturn - value).detach()  # NStepReturn - V
        entropy1 = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(distribution1.scale)           # Distribution Entropy terms
        entropy2 = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(distribution2.scale)
        actorLoss = -(logProb1 * advantage + logProb2 * advantage)-ENTROPYWEIGHT*(entropy1+entropy2)

        # Total Loss
        total_loss = (actorLoss + criticLoss).mean()

        # Calculate Gradients for Back Propagation
        self.opt.zero_grad()
        total_loss.backward()
        
        actorGradients = [param.grad.data for param in self.network.actor.parameters()]
        criticGradients = [param.grad.data for param in self.network.critic.parameters()]

        return actorGradients, criticGradients

    def run(self):
        totalstep = 0
        rewardList = []
        totalReward = 0
        totalRewardList = []
        stateList = []
        actionList = []
        saveDirectory = "C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\Bullet_2023\\Data\\NN"
        os.makedirs(saveDirectory, exist_ok=True)
        while self.episodeNum < MAXEPISODE:
            # Episode Start
            self.env.reset()
            state = self.env.initialState
            totalReward = 0
            
            # Run env
            for i in range(MAXSTEP): 
                stateList.append(state)
                stateOld = state
                action = self.network.actor.selectAction(stateOld)
                action = [action[0].item(), action[1].item()]
                state, reward, done = self.env.step(action)
                actionList.append(action)
                
            # End episode if reached MAXSTEP
                if i == MAXSTEP-1:
                    reward += (1-self.env.world.agent.agentTargetDistanceSS/self.env.world.mapScale)
                    done = True
                totalReward += reward
                rewardList.append(reward)
                
                if totalstep%UPDATESTEP == 0 or done:
                    actorAccumulatedGradient = [torch.zeros_like(param) for param in self.network.actor.parameters()]
                    criticAccumulatedGradient = [torch.zeros_like(param) for param in self.network.critic.parameters()]
                    v = self.globalNetwork.critic.forward(state)
                    nStepReturn = v.detach()
                    
            # Update Accumulated Gradient
                    for j in range(len(stateList)-1,-1,-1):
                        nStepReturn = GAMMA*nStepReturn + rewardList[j]  
                        actorGradient, criticGradient = self.calculateGradient(nStepReturn, stateList[j], actionList[j])
                        actorAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(actorAccumulatedGradient, actorGradient)]
                        criticAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(criticAccumulatedGradient, criticGradient)]      
            # Push and Pull
                    self.upload(actorAccumulatedGradient, criticAccumulatedGradient)
                    self.download()
            # Initialize Trajectory Segment
                    rewardList.clear()
                    stateList.clear()
                    actionList.clear()
            # when episode ends during MAXSTEP
                    if done:
                        break     
                totalstep+=1  
            totalRewardList.append(totalReward)
            print(f"Worker {self.id} = Episode : {self.episodeNum}, Total Reward : {totalReward}, Total Step : {self.env.countStep}")
            self.episodeNum+=1
        # Save Results ##33
            if self.id == 0 and self.episodeNum>1 and self.episodeNum%100 == 0:
                # Save Actor&Critic Network per 1000 episode
                actorPath = os.path.join(saveDirectory, f"actor_ep{self.episodeNum-1}.pth")
                criticPath = os.path.join(saveDirectory, f"critic_ep{self.episodeNum-1}.pth")
                torch.save(self.network.actor.state_dict(), actorPath)
                torch.save(self.network.critic.state_dict(), criticPath)
        # save Reward List and Episodic Time Step List       
        Results = pd.DataFrame({'Total Reward': totalRewardList, 'Time Spend':self.env.timeSpend})
        saveDirectory = f"C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\Bullet_2023\\Data\\Results\\Worker{self.id}Result.xlsx"
        Results.to_excel(saveDirectory)

    def runWithGUI(self):
        totalstep = 0
        rewardList = []
        totalReward = 0
        totalRewardList = []
        stateList = []
        actionList = []
        gui = SimpleGUI(self.env.world.agent.agentPos, self.env.world.agent.targetPos)
        saveDirectory = "C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\Bullet_2023\\Data\\NN"
        os.makedirs(saveDirectory, exist_ok=True)
        while self.episodeNum < MAXEPISODE:
            # Episode Start
            self.env.reset()
            state = self.env.initialState
            totalReward = 0
            
            # Run env
            for i in range(MAXSTEP): 
                stateList.append(state)
                stateOld = state
                action = self.network.actor.selectAction(stateOld)
                action = [action[0].item(), action[1].item()]
                state, reward, done = self.env.step(action)
                actionList.append(action)
                gui.update_positions(self.env.world.agent.agentPos, self.env.world.agent.targetPos)
                gui.draw_objects()
                
            # End episode if reached MAXSTEP
                if i == MAXSTEP-1:
                    reward += (1-self.env.world.agent.agentTargetDistanceSS/self.env.world.mapScale)
                    done = True
                totalReward += reward
                rewardList.append(reward)
                
                if totalstep%UPDATESTEP == 0 or done:
                    actorAccumulatedGradient = [torch.zeros_like(param) for param in self.network.actor.parameters()]
                    criticAccumulatedGradient = [torch.zeros_like(param) for param in self.network.critic.parameters()]
                    v = self.globalNetwork.critic.forward(state)
                    nStepReturn = v.detach()
                    
            # Update Accumulated Gradient
                    for j in range(len(stateList)-1,-1,-1):
                        nStepReturn = GAMMA*nStepReturn + rewardList[j]  
                        actorGradient, criticGradient = self.calculateGradient(nStepReturn, stateList[j], actionList[j])
                        actorAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(actorAccumulatedGradient, actorGradient)]
                        criticAccumulatedGradient = [accumulated_grad + new_grad for accumulated_grad, new_grad in zip(criticAccumulatedGradient, criticGradient)]      
            # Push and Pull
                    self.upload(actorAccumulatedGradient, criticAccumulatedGradient)
                    self.download()
            # Initialize Trajectory Segment
                    rewardList.clear()
                    stateList.clear()
                    actionList.clear()
            # when episode ends during MAXSTEP
                    if done:
                        break     
                totalstep+=1  
            totalRewardList.append(totalReward)
            print(f"Worker {self.id} = Episode : {self.episodeNum}, Total Reward : {totalReward}, Total Step : {self.env.countStep}")
            self.episodeNum+=1
        # Save Results ##33
            if self.id == 0 and self.episodeNum>1 and self.episodeNum%100 == 0:
                # Save Actor&Critic Network per 1000 episode
                actorPath = os.path.join(saveDirectory, f"actor_ep{self.episodeNum-1}.pth")
                criticPath = os.path.join(saveDirectory, f"critic_ep{self.episodeNum-1}.pth")
                torch.save(self.network.actor.state_dict(), actorPath)
                torch.save(self.network.critic.state_dict(), criticPath)
        # save Reward List and Episodic Time Step List       
        Results = pd.DataFrame({'Total Reward': totalRewardList, 'Time Spend':self.env.timeSpend})
        saveDirectory = f"C:\\Users\\shann\\Desktop\\PROGRAMMING\\projects\\Python\\Bullet_2023\\Data\\Results\\Worker{self.id}Result.xlsx"
        Results.to_excel(saveDirectory)

class SimpleGUI:
    def __init__(self, agent_position, target_position):
        self.root = tk.Tk()
        self.root.title("Agent and Target Visualization")
        self.canvas = tk.Canvas(self.root, bg="white", height=400, width=400)
        self.canvas.pack()

        self.agent_position = agent_position
        self.target_position = target_position
        self.draw_objects()

    def draw_objects(self):
        self.canvas.delete("all")
        self.canvas.create_oval(self.agent_position[0]-10, self.agent_position[1]-10, 
                                self.agent_position[0]+10, self.agent_position[1]+10, 
                                fill="red", tags="agent")
        self.canvas.create_oval(self.target_position[0]-10, self.target_position[1]-10, 
                                self.target_position[0]+10, self.target_position[1]+10, 
                                fill="green", tags="target")

    def update_positions(self, new_agent_position, new_target_position):
        self.agent_position = new_agent_position
        self.target_position = new_target_position
        self.draw_objects()      

def workerGUI(globalNetwork, workerId, MapNum, Opt):
    print(f"---------------------------------------------Starting Worker {workerId}---------------------------------------------")
    env = SimpleWorldTL.simpleMapEnv(mapNum = MapNum)
    simpleWorker = SimpleEnvWorker(globalNetwork, env, workerId, Opt)
    simpleWorker.runWithGUI()
    print(f"-------------------------------------------------------Work {workerId} Completed-------------------------------------------------------")

def worker(globalNetwork, workerId, MapNum, Opt):
    print(f"---------------------------------------------Starting Worker {workerId}---------------------------------------------")
    env = SimpleWorldTL.simpleMapEnv(mapNum = MapNum)
    simpleWorker = SimpleEnvWorker(globalNetwork, env, workerId, Opt)
    simpleWorker.run()
    print(f"-------------------------------------------------------Work {workerId} Completed-------------------------------------------------------")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn')
    GlobalNetwork = SimpleEnvGlobalNetwork()
    GlobalNetwork.share_memory()
    Opt = SharedAdam(GlobalNetwork.parameters(),lr=LEARNINGRATE, betas=(0.95, 0.999))
    
    process =[]
    
    for i in range(0,16):
        WorkerProcess = mp.Process(target=worker, args = (GlobalNetwork, i, 3, Opt))
        WorkerProcess.start()
        process.append(WorkerProcess)

    for proc in process:
        proc.join()

# def main():
#     env = SimpleWorldTL.simpleMapEnv(mapNum = 4)

#     GlobalNetwork = SimpleEnvGlobalNetwork()
#     Opt = SharedAdam(GlobalNetwork.parameters(),lr=1e-4, betas=(0.95, 0.999))
#     worker = SimpleEnvWorker(GlobalNetwork, env, 0, Opt)

#     worker.run()

# if __name__ == "__main__":
#     main()