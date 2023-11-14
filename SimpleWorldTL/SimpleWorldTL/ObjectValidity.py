import pybullet as p
import time
from pynput import keyboard
import pybullet_data

PhysicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
StartPos = [0, 0, 1]
StartOrientation = p.getQuaternionFromEuler([0,0,0])

PlaneId = p.loadURDF("Plane_20x20.urdf")
WallOrientation = p.getQuaternionFromEuler([0,0,3.14159 / 2])
WallId1 = p.loadURDF("Wall_20x10x1.urdf", [0,20,0])
WallId2 = p.loadURDF("Wall_20x10x1.urdf", [0,-20,0])
WallId3 = p.loadURDF("Wall_20x10x1.urdf", [20,0,0], WallOrientation)
WallId4 = p.loadURDF("Wall_20x10x1.urdf", [-20,0,0], WallOrientation)

ObstacleId1 = p.loadURDF("Agent_Double_Cylinder.urdf", [0,0,0.2])

p.setGravity(0,0,-10)
while True:
    p.stepSimulation()
    time.sleep(1./240)