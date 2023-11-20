import pybullet as p
import time
from pynput import keyboard
import pybullet_data

WALLORIENTATION = p.getQuaternionFromEuler([0,0,3.14159 / 2])
PhysicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath("C:/Users/shann/Desktop/Modeling/URDF")
StartPos = [0, 0, 1]
StartOrientation = p.getQuaternionFromEuler([0,0,0])

planeId = p.loadURDF("Plane_40x40.urdf")
WallId1 = p.loadURDF("Wall_40x1x10.urdf", [0,40,0])
WallId2 = p.loadURDF("Wall_40x1x10.urdf", [0,-40,0])
WallId3 = p.loadURDF("Wall_40x1x10.urdf", [40,0,0], WALLORIENTATION)
WallId4 = p.loadURDF("Wall_40x1x10.urdf", [-40,0,0], WALLORIENTATION)

ObstacleId1 = p.loadURDF("Agent_Double_Cylinder.urdf", [0,7,0.4])
ObstacleId2 = p.loadURDF("Target_Cylinder.urdf", [0,-7,0.2])
ObstacleId3 = p.loadURDF("Obstacle_Cylinder_2x2x4.urdf")
p.setGravity(0,0,-10)
p.changeDynamics(planeId, -1, lateralFriction=1.0)
while True:
    p.stepSimulation()
    time.sleep(1./240)