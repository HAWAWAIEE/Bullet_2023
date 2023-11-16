import pybullet as p
import time
import math
from pynput import keyboard
import pybullet_data 
from pybullet_utils import bullet_client as bc
import numpy as np
import random
import pybullet as p

STATENUM = 28
NUMRAYS = 12
RAYLENGTH = 5.0
MAXDISTANCE = 100
WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])

class LabelManager:
    def __init__(self):
        self.labelsDict = {}

    def addObject(self, id, label):
        self.labelsDict[id] = label

    def getLabel(self, id):
        return self.labelsDict.get(id, None)



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

class Obstacle:
    def __init__(self, BulletClient, URDF, RangeList = None, physicsClientId=None):
        self.baseLocation = [0,0,0]
        self.baseAngle = [0,0,0,1]
        self.rangeList = RangeList        


        self.bulletClient = BulletClient
    
        if physicsClientId == None:
            self.serverId = self.bulletClient._clinet
        else:
            self.serverId = physicsClientId
            
        self.id = self.bulletClient.loadURDF(URDF, self.baseLocation, self.baseAngle, physicsClientId = self.serverId)
        
    def reset(self, randomness = 0, wantedPosition = None):
        if randomness:
            self.baseLocation = randomPosition(self.rangeList[0:3])
            self.baseAngle = randomQuaternionZaxis(self.rangeList[3])
        else:
            self.baseLocation = wantedPosition
        self.bulletClient.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, physicsClientId=self.serverId)
        
class Map:
    def __init__(self, physicsClientId = None):
        self.bulletClient = bc.BulletClient(connection_mode = p.direct)
        self.bulletClient.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
        self.labelManager = LabelManager()
        
        if physicsClientId == None:
            self.serverId = self.bulletClient._clinet
        else:
            self.serverId = physicsClientId
            
        
    def generateSize20x20Map(self):
        # Loading 20x20 Size Map
        planeId = self.bulletClient.loadUrdf("Plane_20x20.urdf")
        self.labelManager.addObject(planeId, 0)
        WallId1 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [0,20,0])
        self.labelManager(WallId1, 1)
        WallId2 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
        self.labelManager(WallId2, 1)
        WallId3 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [20,0,0], WALLORIENTATION)
        self.labelManager(WallId3, 1)
        WallId4 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WALLORIENTATION)
        self.labelManager(WallId4, 1)
        
        # Set Basic Environmental Parameters
        self.bulletClient.setGravity(0, 0, -10, physicsClientId=self.serverId)
        
    def generateSize40x40Map(self):
        # Loading 20x20 Size Map
        planeId = self.bulletClient.loadUrdf("Plane_40x40.urdf")
        self.labelManager.addObject(planeId, 0)

    def simpleMap01(self):      
        # Loading Obstacles
        obstacle1 = Obstacle()
        
        # Loading Agent
        
                
        pass
    pass

class Env:
    def __init__(self):
        
        pass
    def step(self):
        
        pass
    def reset(self):
        

        pass
    def 
        
        
        
        
        
        pass