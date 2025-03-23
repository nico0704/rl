from ppo import PPO
from race_car_env import RaceCarEnv

def test():
    print("============================================================================================")
    env_name = "test"
    has_continuous_action_space = True
    max_ep_len = 1000
    action_std = 0.1

    render = True
    frame_delay = 0

    total_test_episodes = 10

    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    env = RaceCarEnv()
    state_dim = 7
    action_dim = 2
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    random_seed = 0
    run_num_pretrained = 0
    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    ppo_agent.load(checkpoint_path)
    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break

        ppo_agent.buffer.clear()
        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

if __name__ == '__main__':
    test()