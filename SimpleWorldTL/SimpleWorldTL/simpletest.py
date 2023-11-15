import pybullet as p
import time
import random
import math
from pynput import keyboard
import pybullet_data 
from pybullet_utils import bullet_client as bc
import numpy as np


STATENUM = 28
NUMRAYS = 12
RAYLENGTH = 5.0
MAXDISTANCE = 100
WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])
RAYEXCLUDE = 0b0001
RAYMASK = 0b1110
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
                label = "None"
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = "Unknown"

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
                label = "None"
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = "Unknown"

            self.sensorData[2*i+4] = label
            self.sensorData[2*i+5] = hitDistance

    def raycastBatchWithLabelsFromLinkFixed(self, LinkIndex):
        sensorLinkPos = np.tile(self.bulletClient.getLinkState(self.id, LinkIndex)[0], (12,1))
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
                label = "None"
            else:
                label = self.labelManager.getLabel(hitObjectId)
                if label is None:
                    label = "Unknown"

            self.sensorData[2*i+4] = label
            self.sensorData[2*i+5] = hitDistance

    def relativeDirection(self):
        self.targetPos[0:2] = self.bulletClient.getBasePositionAndOrientation(self.targetId)[0][0:2]
        self.agentPos[0:2] = self.bulletClient.getBasePositionAndOrientation(self.id)[0][0:2]
        self.relativeLocation = self.targetPos - self.agentPos
        self.agentTargetDistance = min(np.dot(self.relativeLocation, self.relativeLocation), MAXDISTANCE)
        self.sensorData[2:4] = np.round(self.relativeLocation/math.sqrt(self.agentTargetDistance),3)

    def reset(self, randomness, wantedPosition = None):
        self.bulletClient.resetBaseVelocity(self.id, linearVelocity = 0, angularVelocity = 0, physicsClientId = self.serverId)
        if randomness:
            self.baseLocation = randomPosition(self.rangeList[0:3])
            self.baseAngle = randomQuaternionZaxis(self.rangeList[3])
        else:
            self.baseLocation = wantedPosition
        self.bulletClient.resetBasePositionAndOrientation(self.id, posObj = self.baseLocation, ornObj = self.baseAngle, physicsClientId = self.serverId)       
    def observation(self):
        self.raycastBatchWithLabels()
        self.relativeDirection()
        self.sensorData[0:2] = np.round(self.bulletClient.getBaseVelocity(self.id, self.serverId)[0][0:2],3)
        print(self.sensorData)
        
class Obstacle:
    def __init__(self, URDF, BulletClient, RangeList = None, physicsClientId=None):
        self.bulletClient = BulletClient
        self.id = self.bulletClient.loadURDF(URDF)

        self.baseLocation = [0,0,0]
        self.baseAngle = [0,0,0,1]
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
    Class for Making Map
    Follow the following order
    1. Make Map Instance
    2. Choose Map Size
    3. Choose Map
    4. Use Map Reset
    """
    def __init__(self, physicsClientId = None):
        self.bulletClient = bc.BulletClient(connection_mode = p.GUI)
        self.bulletClient.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
        self.labelManager = LabelManager()
        if physicsClientId == None:
            self.serverId = self.bulletClient._client
        else:
            self.serverId = physicsClientId
        self.rangeListList = []
        self.target = None
        self.agent = None

    def generateSize20x20Map(self):
        # Loading 20x20 Size Map
        planeId = self.bulletClient.loadURDF("Plane_20x20.urdf")
        self.labelManager.addObject(planeId, 0)
        WallId1 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [0,20,0])
        self.labelManager.addObject(WallId1, 1)
        WallId2 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
        self.labelManager.addObject(WallId2, 1)
        WallId3 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [20,0,0], WALLORIENTATION)
        self.labelManager.addObject(WallId3, 1)
        WallId4 = self.bulletClient.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WALLORIENTATION)
        self.labelManager.addObject(WallId4, 1)        
    
    def generateSize40x40Map(self):
        # Loading 40x40 Size Map
        pass

    # Functions for simpleMap01
    def simpleMap01(self):
        self.rangeListList = [[[-18,18],[6,18],[0.4,0.4],[0,0]],[[-18,18],[-6,-18],[0.4,0.4],[0,0]]]

        # Loading Obstacles
        obstacle1 = Obstacle("Cube_12x2x2.urdf", self.bulletClient)
        # Loading Target
        self.target = Target("Target_Cylinder.urdf", self.bulletClient, self.rangeListList[0])
        # Loading Agent
        self.agent = Agent("Agent_Double_Cylinder.urdf", self.target.id, self.bulletClient, self.labelManager, self.rangeListList[1])
        
    def simpleMap01Reset(self):
        # Randomly select rangeList for agent and target
        [self.agent.rangeList, self.target.rangeList] = random.sample(self.rangeListList,2)
        # Reset Agent and Target
        self.agent.reset(1)
        self.target.reset(1)
        
    # Functions for simpleMap02
    def simpleMap02(self):
        self.rangeListList = []

    pass


worker1 = Map()
worker1.generateSize20x20Map()
worker1.simpleMap01()
worker1.simpleMap01Reset()

while True:
    p.stepSimulation()
    time.sleep(1./240)