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

Agent = p.loadURDF("Agent_Double_Cylinder.urdf", [1,10,0.4])
ObjectId1 = p.loadURDF("Obstacle_Concave.urdf")
[[6,18],[-18,18],[0.4,0.4],[0,0]]


p.setGravity(0,0,-10)
while True:
    p.stepSimulation()
    time.sleep(1./240)