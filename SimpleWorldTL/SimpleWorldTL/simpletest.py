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
PhysicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])

# Loading Objects
planeId = p.loadURDF("Plane_20x20.urdf")
WallId1 = p.loadURDF("Wall_20x10x1.urdf", [0,20,0])

WallId2 = p.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
WallId3 = p.loadURDF("Wall_20x10x1.urdf", [20,0,0], WALLORIENTATION)
WallId4 = p.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WALLORIENTATION)

# Loading Obstacles
ObstacleId1 = p.loadURDF("Obstacle_Cube_4x2x2.urdf", [3,3,2])
# Loading Target
TargetId = p.loadURDF("Cylinder_2x2x4.urdf", [10,10,4])
# Loading Agent
TesterId = p.loadURDF("Cylinder_1x1x1.urdf", startPos, startOrientation)

controller = HeuristicAgentController(TesterId)


p.setGravity(0,0,-10)

while True:
    p.stepSimulation()
    time.sleep(1./240)