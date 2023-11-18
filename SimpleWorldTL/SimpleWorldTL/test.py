import pybullet as p
from pybullet_utils import bullet_client
import torch.multiprocessing as mp
import time

def start_simulation():
    
    bullet_server = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY_SERVER)
    bullet_server.setGravity(0, 0, -9.81)
    while True:
        bullet_server.stepSimulation()
        time.sleep(1./240)

def connect_to_gui():
    
    p.connect(p.SHARED_MEMORY_GUI)
    while True:
        p.stepSimulation()
        time.sleep(1./240)

if __name__ == '__main__':
    
    simulation_process = mp.Process(target=start_simulation)
    simulation_process.start()

    time.sleep(3)
    connect_to_gui()

    simulation_process.join()
