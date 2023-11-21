from re import S
import time
import random
import math
from pynput import keyboard
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data 
from pybullet_utils import bullet_client as bc
from utils import (SB3ToTorchNN, nnKeyChanger)
import torch


STATENUM = 20
NUMRAYS = 8
RAYLENGTH = 10.0
MAXDISTANCE = 400
WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])
RAYEXCLUDE = 0b0001
RAYMASK = 0b1110

STEPTIME = 60
MAXSTEP = 2000
DPBACOEF = 0.1
POTENTIALCOEF = 1


POLICYPATH = r"C:\Users\shann\Desktop\PROGRAMMING\Python\Past_Results\BigEnv_Normal_16workers_4map_10000000timesteps_Results\nn\policy.pth"

def randomQuaternionZaxis(RangeList):
    """
    Generate a random quaternion representing a rotation around the Z-axis,
    with an angle between 'a' and 'b' degrees using PyBullet.

    Parameters:
    RangeList : [a,b]

    Returns:
    Quaternion (x, y, z, w) representing the rotation.
    """
    return p.getQuaternionFromEuler([0, 0, np.radians(random.uniform(RangeList[0], RangeList[1]))])

def randomPosition(RangeList):
    """
    Generate a random position (x,y,z) in range [[a,b][c,d][e,f]]
    
    Prameters:
    RangeList : [[a,b][c,d][e,f]]
    
    Returns:
    Coordinate (x,y,z) representing the random position
    """
    return [random.uniform(RangeList[0][0], RangeList[0][1]), random.uniform(RangeList[1][0], RangeList[1][1]), random.uniform(RangeList[2][0], RangeList[2][1])]

# Heuristic Controller Class For Environment Test
class HeuristicAgentController:
    def __init__(self, agent_id, BulletClient):
        # Basic Setting for Heuristic Control
        self.agent_id = agent_id
        self.linear_velocity = 5
        self.angular_velocity = 4
        self.p = BulletClient

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        try:
            if key.char == '1':
                self.move_forward()
            elif key.char == '2':
                self.move_backward()
            elif key.char == '3':
                self.turn_left()
            elif key.char == '4':
                self.turn_right()
        except AttributeError:
            pass

    def move_forward(self):
        _, orientation = self.p.getBasePositionAndOrientation(self.agent_id)
        forwardVec = self.p.getMatrixFromQuaternion(orientation)[0:3]
        forwardVec = [self.linear_velocity * f for f in forwardVec]
        self.p.resetBaseVelocity(self.agent_id, linearVelocity=forwardVec)

    def move_backward(self):
        _, orientation = self.p.getBasePositionAndOrientation(self.agent_id)
        forwardVec = self.p.getMatrixFromQuaternion(orientation)[0:3]
        backwardVec = [-self.linear_velocity * f for f in forwardVec]
        self.p.resetBaseVelocity(self.agent_id, linearVelocity=backwardVec)

    def turn_left(self):
        self.p.resetBaseVelocity(self.agent_id, angularVelocity=[0, 0, self.angular_velocity])

    def turn_right(self):
        self.p.resetBaseVelocity(self.agent_id, angularVelocity=[0, 0, -self.angular_velocity])
    
# Class for Identifying Labels
# Every Objects Should be added to Label Manager when loaded
class LabelManager:
    def __init__(self):
        self.labelsDict = {}

    def addObject(self, id, label):
        self.labelsDict[id] = label

    def getLabel(self, id):
        return self.labelsDict.get(id, None)

# Agent Class
class Agent:
    def __init__(self, URDF:str, TargetId:int, BulletClient:object, labelManager:object, RangeList:list, physicsClientId:int = None):
        self.baseLocation = [0,0,1]
        self.baseAngle = [0,0,0,1]
        self.rangeList = RangeList

        self.bulletClient = BulletClient
        self.id = self.bulletClient.loadURDF(URDF, self.baseLocation, self.baseAngle)

        self.targetId = TargetId
        self.serverId = physicsClientId
        self.sensorData = [0]*STATENUM
        self.bulletClient = BulletClient
        self.labelManager = labelManager
            
        if physicsClientId == None:
            self.serverId = self.bulletClient._client
        else:
            self.serverId = physicsClientId

        self.rayOrigin = [[0, 0, 0] for _ in range(NUMRAYS)]
        self.rayTarget = [[0, 0, 0] for _ in range(NUMRAYS)]
        
        self.agentPos = np.array([0,0])
        self.targetPos = np.array([0,0])
        self.relativeLocation = np.array([0,0])
        self.agentTargetDistance = 0
        self.agentTargetDistanceSS = 0
        self.relativeLocationRatio = 0

    def raycastBatchWithLabels(self):
        for i in range(NUMRAYS):
            angle = (math.pi * 2 / NUMRAYS) * i
            dx = RAYLENGTH * math.cos(angle)
            dy = RAYLENGTH * math.sin(angle)
            dz = 0
            self.rayTarget[i] = [dx, dy, 0]

        rayResults = self.bulletClient.rayTestBatch(self.rayOrigin, self.rayTarget, parentObjectUniqueId=self.id, physicsClientId = self.serverId)
    
        for i,result in enumerate(rayResults):
            hitObjectId, hitFraction = result[0], result[2]
            hitDistance = round(hitFraction * RAYLENGTH,3)

            if hitObjectId == -1:
                label = 0
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = 6

            self.sensorData[2*i+4] = label
            self.sensorData[2*i+5] = hitDistance
            
    def raycastBatchWithLabelsFromLink(self, LinkIndex):
        sensorLinkPos = np.array(self.bulletClient.getLinkState(self.id, LinkIndex)[0])
        for i in range(NUMRAYS):
            angle = (math.pi * 2 / NUMRAYS) * i
            dx = RAYLENGTH * math.cos(angle)
            dy = RAYLENGTH * math.sin(angle)
            dz = 0
            self.rayTarget[i] = np.array([dx, dy, 0]) + sensorLinkPos

        rayResults = self.bulletClient.rayTestBatch(sensorLinkPos, self.rayTarget, parentObjectUniqueId=self.id, physicsClientId = self.serverId)
    
        for i,result in enumerate(rayResults):
            hitObjectId, hitFraction = result[0], result[2]
            hitDistance = round(hitFraction * RAYLENGTH,3)

            if hitObjectId == -1:
                label = 0
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = 6

            self.sensorData[2*i+4] = label
            self.sensorData[2*i+5] = hitDistance

    def raycastBatchWithLabelsFromLinkFixed(self, LinkIndex):
        sensorLinkPos = np.tile(self.bulletClient.getLinkState(self.id, LinkIndex)[0], (NUMRAYS,1))
        for i in range(NUMRAYS):
            angle = (math.pi * 2 / NUMRAYS) * i
            dx = RAYLENGTH * math.cos(angle)
            dy = RAYLENGTH * math.sin(angle)
            dz = 0
            self.rayTarget[i] = np.array([dx, dy, 0]) + sensorLinkPos[i]

        rayResults = self.bulletClient.rayTestBatch(sensorLinkPos, self.rayTarget, physicsClientId = self.serverId, collisionFilterMask = RAYMASK)
    
        for i,result in enumerate(rayResults):
            hitObjectId, hitFraction = result[0], result[2]
            hitDistance = round(hitFraction * RAYLENGTH,3)

            if hitObjectId == -1:
                label = 1
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = 6

            self.sensorData[2*i] = label
            self.sensorData[2*i+1] = hitDistance

    def relativeDirection(self):
        self.targetPos[0:2] = self.bulletClient.getBasePositionAndOrientation(self.targetId)[0][0:2]
        self.agentPos[0:2] = self.bulletClient.getBasePositionAndOrientation(self.id)[0][0:2]
        self.relativeLocation = self.targetPos - self.agentPos
        self.agentTargetDistanceSS = np.dot(self.relativeLocation, self.relativeLocation)
        self.agentTargetDistance = min(self.agentTargetDistanceSS, MAXDISTANCE)+0.000001
        self.sensorData[16:18] = np.round(self.relativeLocation/math.sqrt(self.agentTargetDistance),3)
        self.sensorData[18:20] = np.round(self.agentPos,3)

    def reset(self, randomness, wantedPosition = None):
        self.bulletClient.resetBaseVelocity(self.id, linearVelocity = 0, angularVelocity = 0, physicsClientId = self.serverId)
        if randomness:
            self.baseLocation = randomPosition(self.rangeList[0:3])
            self.baseAngle = randomQuaternionZaxis(self.rangeList[3])
        else:
            self.baseLocation = wantedPosition
        self.bulletClient.resetBasePositionAndOrientation(self.id, posObj = self.baseLocation, ornObj = self.baseAngle, physicsClientId = self.serverId)
        self.observation()
        
    def observation(self):
        self.raycastBatchWithLabelsFromLinkFixed(0)
        self.relativeDirection()
        # self.sensorData[0] = np.round(self.bulletClient.getBaseVelocity(self.id)[0][0:2],3)
        
class Obstacle:
    def __init__(self, URDF, BulletClient, BaseLocation= [0,0,0], BaseAngle = [0,0,0,1], RangeList = None, physicsClientId=None):
        self.bulletClient = BulletClient
        self.id = self.bulletClient.loadURDF(URDF, BaseLocation, BaseAngle)

        self.baseLocation = BaseLocation
        self.baseAngle = BaseAngle
        self.rangeList = RangeList
    
        if physicsClientId == None:
            self.serverId = self.bulletClient._client
        else:
            self.serverId = physicsClientId
        
    def reset(self, randomness = 0, wantedPosition = None):
        if randomness:
            self.baseLocation = randomPosition(self.rangeList[0:3])
            self.baseAngle = randomQuaternionZaxis(self.rangeList[3])
        else:
            pass
        self.bulletClient.resetBasePositionAndOrientation(self.id, posObj = self.baseLocation, ornObj = self.baseAngle, physicsClientId = self.serverId)

class Target:
    def __init__(self, URDF:str, BulletClient:object, RangeList:list = None, physicsClientId:int=None):
        self.baseLocation = [0,0,1]
        self.baseAngle = [0,0,0,1]
        self.rangeList = RangeList
        
        self.bulletClient = BulletClient
        self.id = self.bulletClient.loadURDF(URDF, self.baseLocation, self.baseAngle)
    
        if physicsClientId == None:
            self.serverId = self.bulletClient._client
        else:
            self.serverId = physicsClientId
        
    def reset(self, randomness = 0, wantedPosition = None):
        if randomness:
            self.baseLocation = randomPosition(self.rangeList[0:3])
        else:
            self.baseLocation = wantedPosition
        self.bulletClient.resetBasePositionAndOrientation(self.id, posObj = self.baseLocation, ornObj = self.baseAngle, physicsClientId = self.serverId)

        
class Map:
    """
    Class for Making BigMap
    Follow the following order
    1. Make Map Instance
    2. Choose Map Size (40x40, 80x80)
    3. Choose Map (Currently only one map each Map Size available)
    4. Use Map Reset
    """
    def __init__(self, physicsClientId:int = None):
        self.bulletClient = bc.BulletClient(connection_mode = p.DIRECT)
        self.bulletClient.setGravity(0,0,-10)
        self.bulletClient.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
        self.labelManager = LabelManager()
        if physicsClientId == None:
            self.serverId = self.bulletClient._client
        else:
            self.serverId = physicsClientId
        self.rangeListList = []
        self.target = None
        self.agent = None
        self.mapScale = 0
        self.mapRadius = 0
        # Set FPS
        self.bulletClient.setTimeStep(1/60)

    def generateSize40x40Map(self):
        """
        Function for loading 40x40 SIze Map
        Map Scale = 1600
        Map Radius = 40
        """
        # Loading 40x40Size Map
        planeId = self.bulletClient.loadURDF("Plane_20x20.urdf")
        self.labelManager.addObject(planeId, 4)
        WallId1 = self.bulletClient.loadURDF("Wall_20x1x10.urdf", [20,0,0])
        self.labelManager.addObject(WallId1, 2)
        WallId2 = self.bulletClient.loadURDF("Wall_20x1x10.urdf", [-20,0,0])
        self.labelManager.addObject(WallId2, 2)
        WallId3 = self.bulletClient.loadURDF("Wall_20x1x10.urdf", [0,20,0], WALLORIENTATION)
        self.labelManager.addObject(WallId3, 2)
        WallId4 = self.bulletClient.loadURDF("Wall_20x1x10.urdf", [0,-20,0], WALLORIENTATION)
        self.labelManager.addObject(WallId4, 2) 
        self.mapScale = 1600
        self.mapRadius = 40
        
    def generateSize80x80Map(self):
        """
        Function for loading 80x80 Size Map
        Map Scale = 6400
        Map Radius = 80
        """
        # Loading 80x80 Size Map
        planeId = self.bulletClient.loadURDF("Plane_40x40.urdf")
        self.labelManager.addObject(planeId, 4)
        WallId1 = self.bulletClient.loadURDF("Wall_40x1x5.urdf", [40,0,0])
        self.labelManager.addObject(WallId1, 2)
        WallId2 = self.bulletClient.loadURDF("Wall_40x1x5.urdf", [-40, 0,0])
        self.labelManager.addObject(WallId2, 2)
        WallId3 = self.bulletClient.loadURDF("Wall_40x1x5.urdf", [10,0,0], WALLORIENTATION)
        self.labelManager.addObject(WallId3, 2)
        WallId4 = self.bulletClient.loadURDF("Wall_40x1x5.urdf", [-10,0,0], WALLORIENTATION)
        self.labelManager.addObject(WallId4, 2) 
        self.mapScale = 6400
        self.mapRadius = 80
        
    # Functions for BigMap01
    def BigMap01(self):
        """
        BigMap01 for 40x40 Size Map
        9 rangeList for Target&Agent Spawn location
        Many Obstacles
        """
        self.rangeListList = [[[13,17],[13,17],[0.25,0,25],[0,0]],[[-17,-13],[13,17],[0.25,0.25],[0,0]],[[13,17],[-17,-13],[0.25,0.25],[0,0]],[[-17,-13],[-17,-13],[0.25,0.25],[0,0]]]
        # Loading Obstacles
        obstacle1 = Obstacle("Obstacle_Cube_1x6x4.urdf", self.bulletClient)
        self.labelManager.addObject(obstacle1.id, 3)
        obstacle2 = Obstacle("Obstacle_Cube_1x6x4.urdf", self.bulletClient, [0,0,0], WALLORIENTATION)
        self.labelManager.addObject(obstacle2.id, 3)
        obstacle3 = Obstacle("Obstacle_Cylinder_2x2x4.urdf", self.bulletClient, [10,10,0])
        self.labelManager.addObject(obstacle3.id, 3)
        obstacle4 = Obstacle("Obstacle_Cylinder_2x2x4.urdf", self.bulletClient, [-10,10,0])
        self.labelManager.addObject(obstacle4.id, 3)
        obstacle5 = Obstacle("Obstacle_Cylinder_2x2x4.urdf", self.bulletClient, [10,-10,0])
        self.labelManager.addObject(obstacle5.id, 3)
        obstacle6 = Obstacle("Obstacle_Cylinder_2x2x4.urdf", self.bulletClient, [-10,-10,0])
        self.labelManager.addObject(obstacle6.id, 3)       
        obstacle7 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [0, 10, 0])
        self.labelManager.addObject(obstacle7.id, 3)
        obstacle8 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [0, 15, 0])
        self.labelManager.addObject(obstacle8.id, 3)
        obstacle9 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [0, -10, 0])
        self.labelManager.addObject(obstacle9.id, 3)
        obstacle10 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [0, -15, 0])
        self.labelManager.addObject(obstacle10.id, 3)
        obstacle11 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [10, 0, 0])
        self.labelManager.addObject(obstacle11.id, 3)
        obstacle12 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [15, 0, 0])
        self.labelManager.addObject(obstacle12.id, 3)
        obstacle13 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [-10, 0, 0])
        self.labelManager.addObject(obstacle13.id, 3)
        obstacle14 = Obstacle("Obstacle_Cube_1x1x4.urdf", self.bulletClient, [-15, 0, 0])
        self.labelManager.addObject(obstacle14.id, 3)
        
        # Loading Target
        self.target = Target("Target_Cylinder.urdf", self.bulletClient, self.rangeListList[0])
        self.labelManager.addObject(self.target.id, 5)
        # Loading Agent
        self.agent = Agent("Agent_Double_Cylinder.urdf", self.target.id, self.bulletClient, self.labelManager, self.rangeListList[1], physicsClientId=None)
        self.labelManager.addObject(self.agent.id, 9)
        
    def BigMap01Reset(self):
        # Randomly select rangeList for agent and target
        [self.agent.rangeList, self.target.rangeList] = random.sample(self.rangeListList,2)
        # Reset Agent and Target
        self.agent.reset(1)
        self.target.reset(1)
        
    # Functions for BigBigMap01
    def BigBigMap01(self):
        self.rangeListList = [[[-7,7],[3,7],[0.25,0.25],[0,0]],[[-7,7],[-3,-7],[0.25,0.25],[0,0]]]
        # Loading Obstacles
        obstacle1 = Obstacle("Obstacle_Cube_4x1x2.urdf", self.bulletClient)
        self.labelManager.addObject(obstacle1.id, 3)
        # Loading Target
        self.target = Target("Target_Cylinder.urdf", self.bulletClient, self.rangeListList[0])
        self.labelManager.addObject(self.target.id, 5)
        # Loading Agent
        self.agent = Agent("Agent_Double_Cylinder.urdf", self.target.id, self.bulletClient, self.labelManager, self.rangeListList[1], physicsClientId=None)
        self.labelManager.addObject(self.agent.id, 9)
    def BigBigMap01Reset(self):
        # Randomly select rangeList for agent and target
        [self.agent.rangeList, self.target.rangeList] = random.sample(self.rangeListList,2)
        # Reset Agent and Target
        self.agent.reset(1)
        self.target.reset(1)
        
class bigMapEnv(gym.Env):
    def __init__(self, mapNum:int):
        super().__init__()
        # Define Observation Space and Action Space
        self.observation_space = spaces.Box(low=-100, high=100, shape=(STATENUM,), dtype=np.float32)
        self.action_space = spaces.Box(low = -2, high = 2, shape=(2,), dtype = np.float32)
        self.done = False
        # Basic World configuration
        self.mapNum = mapNum
        if mapNum == 1:
            # Generate BigMap01 world
            self.world = Map()
            self.world.generateSize40x40Map()
            self.world.BigMap01()      
            self.world.BigMap01Reset()    
            
        else:
            # Generate BigBigMap01 world
            self.world = Map()
            self.world.generateSize80x80Map()
            self.world.BigBigMap01()      
            self.world.BigBigMap01Reset()
            
        """
        Following Properties are for Recording Episodic Results
        1. Time Spend in Simulation per Episode
        """
        
        # Set Initial State
        self.initialState = self.world.agent.sensorData 
        self.initialDis = self.world.agent.agentTargetDistanceSS
        self.info = {}
        # method for detecting time
        self.countStep = 0
        self.timeSpend = []
        print(f"------------------Map : {self.mapNum}---ID : {self.world.serverId}------------------")


    def step(self, action):
        action = [max(-2, min(x, 2)) for x in action]
        # Perform Action. Change x/y velocity with action
        self.world.bulletClient.resetBaseVelocity(self.world.agent.id, linearVelocity = [action[0], action[1],0])
        self.world.agent.observation()
        observation = self.world.agent.sensorData
        observation[18] = observation[18]/self.world.mapRadius
        observation[19] = observation[19]/self.world.mapRadius
        # Determine how much time will 'a step' takes
        # Determine Reward and Done
        for i in range(STEPTIME):
            self.world.bulletClient.stepSimulation()
            if self.targetCollision():
                self.done = self.targetCollision()
                break     
        self.countStep += 1
        
        reward = 4 if self.done else -0.001 
        if self.countStep >= MAXSTEP:
            reward += (1-self.world.agent.agentTargetDistanceSS/self.world.mapScale)
            self.done = True
            truncated = True
        else:
            truncated = False

        return observation, reward, self.done, truncated, self.info
        
    def reset(self, seed = None):
        if self.mapNum == 1:
            self.world.BigMap01Reset()
        else:
            self.world.BigBigMap01Reset()
        # Set Initial State
        self.initialState = self.world.agent.sensorData 
        self.initialDis = self.world.agent.agentTargetDistanceSS
        # Save and Reset Time
        self.timeSpend.append(self.countStep*STEPTIME)
        self.countStep = 0
        self.done = False
        return self.initialState, {}

    # Collision Detection Logic
    def targetCollision(self):
        contacts = self.world.bulletClient.getContactPoints(bodyA=self.world.agent.id, bodyB=self.world.target.id)
        if len(contacts) > 0:
            return True
        else:
            return False
        
class bigMapEnvDPBA(gym.Env):
    def __init__(self, mapNum:int):
        super().__init__()
        # Define Observation Space and Action Space
        self.observation_space = spaces.Box(low=-100, high=100, shape=(STATENUM,), dtype=np.float32)
        self.action_space = spaces.Box(low = -2, high = 2, shape=(2,), dtype = np.float32)

        # Load Expert Actor-Critic from POLICYPATH
        expert_state_dict = nnKeyChanger(torch.load(POLICYPATH, map_location=torch.device('cpu')))
        self.expert = SB3ToTorchNN(STATENUM, 2)
        self.expert = self.expert.load_state_dict(expert_state_dict)
        # Basic World configuration
        self.mapNum = mapNum
        if mapNum == 1:
            # Generate BigMap01 world
            self.world = Map()
            self.world.generateSize40x40Map()
            self.world.BigMap01()      
            self.world.BigMap01Reset()    
            
        else:
            # Generate BigBigMap01 world
            self.world = Map()
            self.world.generateSize80x80Map()
            self.world.BigBigMap01()      
            self.world.BigBigMap01Reset()
        """
        Following Properties are for Recording Episodic Results
        1. Time Spend in Simulation per Episode
        """
        
        # Set Initial State
        self.done = False
        self.initialState = self.world.agent.sensorData 
        self.initialDis = self.world.agent.agentTargetDistanceSS
        self.info = {}
        # method for detecting time
        self.countStep = 0
        self.timeSpend = []
        print(f"------------------Map : {self.mapNum}---ID : {self.world.serverId}------------------")

    def dpbaReward(self, oldObs, newObs):
        # Normalize / Clamp State
        # Add Later
        auxReward = DPBACOEF*(POTENTIALCOEF*self.expert.valueForward(newObs)-self.expert.valueForward(oldObs))
        return auxReward

    def step(self, action):
        action = [max(-2, min(x, 2)) for x in action]
        # Perform Action. Change x/y velocity with action
        self.world.bulletClient.resetBaseVelocity(self.world.agent.id, linearVelocity = [action[0], action[1],0])
        oldObs  = self.world.agent.sensorData
        self.world.agent.observation()
        observation = self.world.agent.sensorData
        observation[18] = observation[18]/self.world.mapRadius
        observation[19] = observation[19]/self.world.mapRadius
        # Determine how much time will 'a step' takes
        # Determine Reward and Done
        for i in range(STEPTIME):
            self.world.bulletClient.stepSimulation()
            if self.targetCollision():
                self.done = self.targetCollision()
                break     
        self.countStep += 1
        
        reward = 4 if self.done else -0.001 + self.dpbaReward(oldObs,observation)
        if self.countStep >= MAXSTEP:
            reward += (1-self.world.agent.agentTargetDistanceSS/self.world.mapScale)
            self.done = True
            truncated = True
        else:
            truncated = False

        return observation, reward, self.done, truncated, self.info
        
    def reset(self, seed = None):
        if self.mapNum == 1:
            self.world.BigMap01Reset()
        else:
            self.world.BigBigMap01Reset()
        # Set Initial State
        self.initialState = self.world.agent.sensorData 
        self.initialDis = self.world.agent.agentTargetDistanceSS
        # Save and Reset Time
        self.timeSpend.append(self.countStep*STEPTIME)
        self.countStep = 0
        self.done = False
        return self.initialState, {}

    # Collision Detection Logic
    def targetCollision(self):
        contacts = self.world.bulletClient.getContactPoints(bodyA=self.world.agent.id, bodyB=self.world.target.id)
        if len(contacts) > 0:
            return True
        else:
            return False