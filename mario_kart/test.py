#### Test Script for PPO Agent on RaceCarEnv ####
# This script was adapted from:
# https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master
# 
# Usage:
#     python test.py  # Loads the latest run automatically
#     python test.py --run_name ppo_2025-03-25
#     python test.py --run_name my_run --num_episodes 5
# Arguments:
#    --run_name        Name of the run folder to load (inside runs/). If omitted, the latest is used.
#    --num_episodes    Number of episodes to run for testing (default: 10)

import argparse
import os
from ppo import PPO
from env.environment import RaceCarEnv
from utils import load_config, get_latest_run_name


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=False, help="Name of the training run.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes.")
    args = parser.parse_args()
    run_name = args.run_name or get_latest_run_name()
    if not args.run_name:
        print(f"\nNo --run_name provided. Using latest run: '{run_name}'")
        print("To test a specific run, use: python test.py --run_name <name>")
    config_path = os.path.join("runs", run_name, "config.yaml")
    config = load_config(config_path)
    num_episodes = args.num_episodes
    print(f"Loaded config from: {config_path}")
    print(f"Testing run: '{run_name}' for {num_episodes} episodes")
    
    action_std = config["action_std"]
    K_epochs = config["K_epochs"]
    eps_clip = config["eps_clip"]
    gamma = config["gamma"]
    lr_actor = config["lr_actor"]
    lr_critic = config["lr_critic"]
    max_ep = config["max_ep"]
    render = config.get("render", True)
    
    env = RaceCarEnv()
    state_dim, action_dim = env.get_dims()
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    directory = os.path.join("runs", run_name)
    checkpoint_path = os.path.join(directory, f"{run_name}.pth")
    ppo_agent.load(checkpoint_path)
    test_running_reward = 0
    
    print("starting to run agent...\n")

    for ep in range(1, num_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(1, max_ep+1):
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
        ppo_agent.buffer.clear()
        test_running_reward +=  ep_reward
        print(f"Episode : {ep} \t\t Reward : {round(ep_reward,2)}")
        ep_reward = 0

    env.close()
    avg_test_reward = test_running_reward / num_episodes
    print(f"average test reward : {str(round(avg_test_reward, 2))}")


if __name__ == '__main__':
    test()