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
RAYEXCLUDE = 0b0001
RAYMASK = 0b1110

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
    
class LabelManager:
    def __init__(self):
        self.labelsDict = {}

    def addObject(self, id, label):
        self.labelsDict[id] = label

    def getLabel(self, id):
        return self.labelsDict.get(id, None)


class Agent:
    def __init__(self, AgentId, PhysicsClientId, TargetId, BulletClient, labelManager, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1]):
        self.id = AgentId
        self.targetId = TargetId
        self.serverId = PhysicsClientId
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

        rayResults = self.bulletClient.rayTestBatch(self.rayOrigin, self.rayTarget, parentObjectUniqueId=self.id)
    
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
            
    def observation(self):
        self.raycastBatchWithLabelsFromLinkFixed(0)
        print(self.sensorData)
        

# Class for Identifying Labels
# Every Objects Should be added to Label Manager when loaded
p = bc.BulletClient(connection_mode = p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])
# Make LabelManager
pLabelManager = LabelManager()
# Loading Objects
planeId = p.loadURDF("Plane_20x20.urdf")
WallId1 = p.loadURDF("Wall_20x10x1.urdf", [0,20,0])
pLabelManager.addObject(WallId1, "Wall_1")
WallId2 = p.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
pLabelManager.addObject(WallId2, "Wall_2")
WallId3 = p.loadURDF("Wall_20x10x1.urdf", [20,0,0], WALLORIENTATION)
pLabelManager.addObject(WallId3, "Wall_3")
WallId4 = p.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WALLORIENTATION)
pLabelManager.addObject(WallId4, "Wall_4")
# Loading Obstacles
ObstacleId1 = p.loadURDF("Obstacle_Cube_4x2x2.urdf", [3,3,2])
pLabelManager.addObject(ObstacleId1, "Obstacle")
# Loading Target
TargetId = p.loadURDF("Cylinder_2x2x4.urdf", [10,10,4])
pLabelManager.addObject(TargetId, "Target")




# Loading Agent
TesterId = p.loadURDF("Agent_Double_Cylinder.urdf", startPos, startOrientation)
pLabelManager.addObject(TesterId, "Agent")
p.setCollisionFilterGroupMask(TesterId, 0, RAYEXCLUDE, 0b1111)
TesterAgent = Agent(AgentId = TesterId, TargetId = TargetId, PhysicsClientId = p._client, BulletClient = p, labelManager = pLabelManager, baseCoordinate = [0,0,0], baseOrientation = [0,0,0,1])

controller = HeuristicAgentController(TesterId, p)

p.setGravity(0,0,-10)

while True:
    p.stepSimulation()
    TesterAgent.observation()
    time.sleep(1./240)