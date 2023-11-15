from re import S
import pybullet as p
import time
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
    def __init__(self, agentId, targetId, BulletClient, labelManager, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId = None):
        self.id = agentId
        self.targetId = targetId
        self.serverId = physicsClientId
        self.sensorData = [0]*STATENUM
        self.bulletClient = BulletClient
        self.labelManager = labelManager

        self.baseLocation = baseCoordinate
        self.baseAngle = baseOrientation

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
        sensorLinkPos = np.array(self.bulletClient.getLinkState(self.id, LinkIndex)[0])
        for i in range(NUMRAYS):
            angle = (math.pi * 2 / NUMRAYS) * i
            dx = RAYLENGTH * math.cos(angle)
            dy = RAYLENGTH * math.sin(angle)
            dz = 0
            self.rayTarget[i] = np.array([dx, dy, 0]) + sensorLinkPos

        rayResults = self.bulletClient.rayTestBatch(sensorLinkPos, self.rayTarget, physicsClientId = self.serverId)
    
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

    def reset(self):
        self.bulletClient.resetBaseVelocity(self.id, [0,0,0], [0,0,0], self.serverId)
        self.bulletClient.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, self.serverId)    

    def observation(self):
        self.raycastBatchWithLabels()
        self.relativeDirection()
        self.sensorData[0:2] = np.round(self.bulletClient.getBaseVelocity(self.id, self.serverId)[0][0:2],3)
        print(self.sensorData)
        
class Obstacle:
    def __init__(self, obstacleId, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId=None, BulletClient = None):
        self.id = obstacleId
        self.serverId = physicsClientId
        self.bulletClient = BulletClient
        self.baseLocation = baseCoordinate
        self.baseAngle = baseOrientation
        
        
    def reset(self):
        self.bulletClient.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, physicsClientId=self.serverId)

class Target:
    def __init__(self, targetId, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId=None, BulletClient = None):
        self.id = targetId
        self.serverId = physicsClientId
        self.baseLocation = baseCoordinate
        self.baseAngle = baseOrientation
        
    def reset(self):
        self.bulletClient.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, physicsClientId=self.serverId)

class Map:
    def __init(self, physicsClientId = None):
        self.serverId = physicsClientId
        self.bulletClient = bc.BulletClient(connection_mode = p.direct)
        self.bulletClient.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
        self.labelManager = LabelManager()
        
    def simpleMap01(self):
        
        # Loading Map
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
        
        # Loading Agent
        agentId = self.bulletClient.loadURDF("Cylinder_1x1x1")
        
        
        
        pass
    pass