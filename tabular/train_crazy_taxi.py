import matplotlib.pyplot as plt
from environment import TaxiEnvironment
from agent import QLearningAgent
from utils import (
    create_city_graph,
    visualize_heatmap, 
    visualize_city, 
    visualize_episode, 
    visualize_episode_with_imgs,
    plot_rewards
)


EPISODES = 100000
VISU_EPISODE = EPISODES / 10 # visualize 10 episodes
VISUALIZE_TRAINING = True
SAVE_Q_TABLE = False
SAVE_Q_TABLE_PATH = f"q_tables/qt_{EPISODES}.txt"
PLOT = False


def train():
    city, positions = create_city_graph()
    visualize_city(city, positions)

    # initialize environment and agent
    env = TaxiEnvironment(city)
    agent = QLearningAgent(env, mode="train")
    
    # visu setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # reward tracking
    total_rewards = []
    cumulative_rewards = []
    cumulative_reward = 0

    # training
    for episode in range(1, EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        positions_trace = []
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            if episode % VISU_EPISODE == 0 and VISUALIZE_TRAINING:
                positions_trace.append(env.taxi_position)
                # visualize_episode(ax, city, positions, env, positions_trace)
                visualize_episode_with_imgs(ax, city, positions, env, positions_trace)
                ax.set_title(f"Training Progress: Episode {episode}")
                plt.pause(0.25)

            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        # update rewards
        total_rewards.append(total_reward)
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)
    
    if PLOT:
        visualize_heatmap(city, env.edge_usage, positions)
        plot_rewards(total_rewards, cumulative_rewards)
        
    if SAVE_Q_TABLE:
        agent.save_q_table(SAVE_Q_TABLE_PATH)

if __name__ == "__main__":
    train()
