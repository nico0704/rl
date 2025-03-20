from race_car_env import RaceCarEnv
import numpy as np

# intialize enironment
env = RaceCarEnv()

# reset environment
state = env.reset()

for _ in range(300):
    action = np.random.uniform(-1, 1, 2) #random actions for now
    state, reward, done = env.step(action)
    env.render()

    if done:
        state = env.reset()


env.close()