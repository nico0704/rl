from race_car_env import RaceCarEnv
from ppo import PPO

env = RaceCarEnv("circle.txt")
ppo_agent = PPO(state_dim=2, action_dim=2)
ppo_agent.train(env, num_episodes=1000)
env.close()
