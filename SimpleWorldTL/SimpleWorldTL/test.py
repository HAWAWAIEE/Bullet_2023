from re import S
import pybullet as p
import time
import math
from pynput import keyboard
import pybullet_data 
import numpy as np

STATENUM = 28
NUMRAYS = 12
RAYLENGTH = 5.0
MAXDISTANCE = 100

# Heuristic Controller Class For Environment Test
class HeuristicAgentController:
    def __init__(self, agent_id):
        # Basic Setting for Heuristic Control
        self.agent_id = agent_id
        self.linear_velocity = 5
        self.angular_velocity = 4

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
        _, orientation = p.getBasePositionAndOrientation(self.agent_id)
        forwardVec = p.getMatrixFromQuaternion(orientation)[0:3]
        forwardVec = [self.linear_velocity * f for f in forwardVec]
        p.resetBaseVelocity(self.agent_id, linearVelocity=forwardVec)

    def move_backward(self):
        _, orientation = p.getBasePositionAndOrientation(self.agent_id)
        forwardVec = p.getMatrixFromQuaternion(orientation)[0:3]
        backwardVec = [-self.linear_velocity * f for f in forwardVec]
        p.resetBaseVelocity(self.agent_id, linearVelocity=backwardVec)

    def turn_left(self):
        p.resetBaseVelocity(self.agent_id, angularVelocity=[0, 0, self.angular_velocity])

    def turn_right(self):
        p.resetBaseVelocity(self.agent_id, angularVelocity=[0, 0, -self.angular_velocity])
     
# Class for Identifying Labels
# Every Objects Should be added to Label Manager when loaded
class LabelManager:
    labelsDict = {}

    def addObject(id, label):
        LabelManager.labelsDict[id] = label

    def getLabel(id):
        return LabelManager.labelsDict.get(id, None)

# Agent Class
class Agent:
    def __init__(self, agentId, targetId, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId = None):
        self.id = agentId
        self.targetId = targetId
        self.serverId = physicsClientId
        self.sensorData = [0]*STATENUM
        
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

        rayResults = p.rayTestBatch(self.rayOrigin, self.rayTarget, parentObjectUniqueId=self.id, physicsClientId = self.serverId)
    
        for i,result in enumerate(rayResults):
            hitObjectId, hitFraction = result[0], result[2]
            hitDistance = hitFraction * RAYLENGTH

            if hitObjectId == -1:
                label = "None"
            else:
                label = LabelManager.getLabel(hitObjectId)
                if label is None:
                    label = "Unknown"

            self.sensorData[2*i+4] = label
            self.sensorData[2*i+5] = hitDistance

    def relativeDirection(self):
        self.targetPos[0:2] = p.getBasePositionAndOrientation(self.targetId)[0][0:2]
        self.agentPos[0:2] = p.getBasePositionAndOrientation(self.id)[0][0:2]
        self.relativeLocation = self.targetPos - self.agentPos
        self.agentTargetDistance = min(np.dot(self.relativeLocation, self.relativeLocation), MAXDISTANCE)
        self.sensorData[2:4] = self.relativeLocation/math.sqrt(self.agentTargetDistance)

    def reset(self):
        p.resetBaseVelocity(self.id, [0,0,0], [0,0,0], self.serverId)
        p.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, self.serverId)    

    def observation(self):
        self.raycastBatchWithLabels()
        self.relativeDirection()
        self.sensorData[0:2] = p.getBaseVelocity(self.id, self.serverId)[0][0:2]
        print(self.sensorData)
        
class Obstacle:
    def __init__(self, obstacleId, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId=None):
        self.id = obstacleId
        self.serverId = physicsClientId
        self.baseLocation = baseCoordinate
        self.baseAngle = baseOrientation
        
    def reset(self):
        p.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, physicsClientId=self.serverId)

class Target:
    def __init__(self, targetId, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1], physicsClientId=None):
        self.id = targetId
        self.serverId = physicsClientId
        self.baseLocation = baseCoordinate
        self.baseAngle = baseOrientation
        
    def reset(self):
        p.resetBasePositionAndOrientation(self.id, self.baseLocation, self.baseAngle, physicsClientId=self.serverId)

PhysicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
WallOrientation = p.getQuaternionFromEuler([0,0,3.14159 / 2])

# Loading Objects
planeId = p.loadURDF("Plane_20x20.urdf")
WallId1 = p.loadURDF("Wall_20x10x1.urdf", [0,20,0])
LabelManager.addObject(WallId1, "Wall")
WallId2 = p.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
LabelManager.addObject(WallId2, "Wall")
WallId3 = p.loadURDF("Wall_20x10x1.urdf", [20,0,0], WallOrientation)
LabelManager.addObject(WallId3, "Wall")
WallId4 = p.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WallOrientation)
LabelManager.addObject(WallId4, "Wall")

# Loading Obstacles
ObstacleId1 = p.loadURDF("Obstacle_Cube_4x2x2.urdf", [3,3,2])
LabelManager.addObject(ObstacleId1, "Obstacle")
ObstacleCube1 = Obstacle(ObstacleId1, [3,3,2], physicsClientId = PhysicsClient)

# Loading Target
TargetId = p.loadURDF("Cylinder_2x2x4.urdf", [10,10,4])
LabelManager.addObject(TargetId, "Target")
Target1 = Target(TargetId, [10,10,4], physicsClientId = PhysicsClient)

# Loading Agent
TesterId = p.loadURDF("CylinderTester_1x1x1.urdf", startPos, startOrientation)
LabelManager.addObject(TesterId, "Agent")
TesterAgent = Agent(TesterId, TargetId, physicsClientId = PhysicsClient)

controller = HeuristicAgentController(TesterId)


p.setGravity(0,0,-10)

while True:
    p.stepSimulation()
    TesterAgent.observation()
    time.sleep(1./240)