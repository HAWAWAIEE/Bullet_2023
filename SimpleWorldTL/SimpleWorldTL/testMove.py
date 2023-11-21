import BigWorldTest20
import SimpleWorldTL20
import pybullet as p
import gymnasium as gym

def check_env_observation(env_name, steps=10):
    env = BigWorldTest20.bigMapEnv(1)
    env.reset()
    
    for step in range(steps):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Observation = {observation}")

        if done:
            env.reset()

if __name__ == "__main__":
    env_name = 'YourEnv'
    check_env_observation(env_name)