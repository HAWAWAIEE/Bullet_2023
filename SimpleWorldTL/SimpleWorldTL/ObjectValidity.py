import pybullet as p
import time
from pynput import keyboard
import pybullet_data

WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])
PhysicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
StartPos = [0, 0, 1]
StartOrientation = p.getQuaternionFromEuler([0,0,0])

PlaneId = p.loadURDF("Plane_10x10.urdf")
WallOrientation = p.getQuaternionFromEuler([0,0,3.14159 / 2])
WallId1 = p.loadURDF("Wall_10x1x5.urdf", [0,10,0])
WallId2 = p.loadURDF("Wall_10x1x5.urdf", [0,-10,0])
WallId3 = p.loadURDF("Wall_10x1x5.urdf", [10,0,0], WallOrientation)
WallId4 = p.loadURDF("Wall_10x1x5.urdf", [-10,0,0], WallOrientation)

ObstacleId1 = p.loadURDF("Agent_Double_Cylinder.urdf", [0,7,2])
ObstacleId1 = p.loadURDF("Target_Cylinder.urdf")
p.setGravity(0,0,-10)
p.changeDynamics(PlaneId, -1, lateralFriction=1.0)
while True:
    p.stepSimulation()
    time.sleep(1./240)