#### Train Script for PPO Agent on RaceCarEnv ####
# Usage:
#     python train.py
# Description:
#     - Loads training configuration from config.py
#     - Initializes the PPO agent and RaceCarEnv
#     - Saves:
#         - Model weights to:      runs/<run_name>/<run_name>.pth
#         - Config file to:        runs/<run_name>/config.yaml

import os
from race_car_env import RaceCarEnv
from ppo import PPO
import config
from utils import save_config


def train():
    run_name = config.run_name
    max_ep = config.max_ep
    max_train_timestemps = config.max_train_timestemps
    
    action_std = config.action_std
    action_std_decay_rate = config.action_std_decay_rate
    min_action_std = config.min_action_std
    action_std_decay_freq = config.action_std_decay_freq
    
    update_timestep = config.update_timestep
    K_epochs = config.K_epochs
    eps_clip = config.eps_clip
    gamma = config.gamma
    lr_actor = config.lr_actor
    lr_critic = config.lr_critic
    
    render_freq = config.render_freq
    print_freq = config.print_freq
    save_model_freq = config.save_model_freq
    render = config.render
    
    directory = os.path.join("runs", run_name)
    if os.path.exists(directory):
        print(f"the path: '{directory}' already exists. Choose another name for your env.")
        return
    os.makedirs(directory)
    checkpoint_path = os.path.join(directory, f"{run_name}.pth")
    print(f"\ntraining environment name : {run_name}")
    print(f"checkpoint path : {checkpoint_path}")
    config_path = os.path.join(directory, "config.yaml")
    print(f"saving config to {config_path}")
    save_config(config_path)
    
    env = RaceCarEnv()
    # TODO get from env:
    state_dim = 7
    action_dim = 2
    
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
    print("training started...\n")
    while time_step <= max_train_timestemps:
        state = env.reset()
        current_ep_reward = 0
        for t in range(1, max_ep + 1):
            if render and i_episode % render_freq == 0:
                env.render()
            
            # select action  
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step +=1
            current_ep_reward += reward
            
            # update ppo
            if time_step % update_timestep == 0:
                ppo_agent.update()
            
            # decay action std of output action distribution
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")
                print_running_reward = 0
                print_running_episodes = 0
            
            # save model weights
            if time_step % save_model_freq == 0:
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")

            if done:
                break
            
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1
        
    env.close()    
    print("training finished!")
        
if __name__ == '__main__':
    train()