from race_car_env import RaceCarEnv
from ppo import PPO

env = RaceCarEnv()
ppo_agent = PPO(state_dim=7, action_dim=2)
ppo_agent.train(env, num_episodes=200)
env.close()
