import networkx as nx
import matplotlib.pyplot as plt
import random

from utils import (
    save_q_table, 
    visualize_heatmap, 
    visualize_city, 
    visualize_episode, 
    visualize_episode_with_imgs,
    plot_rewards
)


#### city graph ####
def create_city_graph():
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1), (2, 4), (2, 5), (3, 6), (5, 6),
        (1, 7), (7, 8), (8, 5), (6, 9), (9, 10), (10, 4),
        (7, 11), (11, 12), (12, 8), (9, 13), (13, 14), (14, 10)
    ]

    positions = {
        1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1), 5: (2, 0), 6: (2, 1),
        7: (-1, 0), 8: (2, -1), 9: (3, 1), 10: (0, 2),
        11: (-2, 0), 12: (-1, -1), 13: (3, 2), 14: (1, 2)
    }

    weights = {
        (1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 1): 1, (2, 4): 2, (2, 5): 4,
        (3, 6): 5, (5, 6): 6, (1, 7): 3, (7, 8): 1, (8, 5): 4, (6, 9): 3,
        (9, 10): 1, (10, 4): 2, (7, 11): 4, (11, 12): 3, (12, 8): 2,
        (9, 13): 3, (13, 14): 4, (14, 10): 1
    }

    city = nx.Graph()
    city.add_edges_from(edges)
    nx.set_edge_attributes(city, weights, 'weight')
    return city, positions


#### environment ####
class TaxiEnvironment:
    def __init__(self, city_graph):
        self.city = city_graph
        self.edge_usage = {edge: 0 for edge in self.city.edges}  # Track edge usage for heatmap
        self.reset()

    def reset(self):
        self.taxi_position = random.choice(list(self.city.nodes))
        self.passenger_status = 'no_passenger'
        self.passenger_start = random.choice(list(self.city.nodes))
        self.passenger_destination = random.choice(list(self.city.nodes))
        return (self.taxi_position, self.passenger_status, self.passenger_start, self.passenger_destination)

    def step(self, action):
        if action in self.city[self.taxi_position]:
            self.edge_usage[tuple(sorted([self.taxi_position, action]))] += 1
            edge_weight = self.city[self.taxi_position][action]['weight']
            self.taxi_position = action
        else:
            return (self.taxi_position, self.passenger_status, self.passenger_start, self.passenger_destination), -10, False  # Invalid action penalty

        reward = -edge_weight  # Movement cost
        done = False

        if self.passenger_status == 'no_passenger' and self.taxi_position == self.passenger_start:
            self.passenger_status = 'has_passenger'
            reward += 10  # Reward for picking up the passenger
        elif self.passenger_status == 'has_passenger' and self.taxi_position == self.passenger_destination:
            reward += 20  # Reward for successful delivery
            done = True

        return (self.taxi_position, self.passenger_status, self.passenger_start, self.passenger_destination), reward, done


#### agent ####
class QLearningAgent:
    def __init__(self, env, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table = self._initialize_q_table(env)

    def _initialize_q_table(self, env):
        q_table = {}
        for node in env.city.nodes:
            for start in env.city.nodes:
                for dest in env.city.nodes:
                    for status in ['no_passenger', 'has_passenger']:
                        state = (node, status, start, dest)
                        q_table[state] = {neighbor: 0 for neighbor in env.city[node]}
        return q_table

    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        return max(self.q_table[state], key=self.q_table[state].get)

    def choose_action_deploy(self, state):
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_future_q - self.q_table[state][action]
        )


def main():
    city, positions = create_city_graph()
    # visualize_city(city, positions)

    # initialize environment and agent
    env = TaxiEnvironment(city)
    agent = QLearningAgent(env)
    
    # visu setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # reward tracking
    total_rewards = []
    cumulative_rewards = []
    cumulative_reward = 0

    # training
    episodes = 10000000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        positions_trace = []
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # if episode == 2000000-1:
            #     positions_trace.append(env.taxi_position)
            #     # visualize_episode(ax, city, positions, env, positions_trace)
            #     visualize_episode_with_imgs(ax, city, positions, env, positions_trace)
            #     ax.set_title(f"Training Progress: Episode {episode}")
            #     plt.pause(0.25)

            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        # update rewards
        total_rewards.append(total_reward)
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)
    
    # heatmap
    visualize_heatmap(city, env.edge_usage, positions)
    # reward plots
    plot_rewards(total_rewards, cumulative_rewards)
    # save q table
    save_q_table(agent)
    
    import time
    time.sleep(1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for _ in range(10):
        # deploy
        state = env.reset()
        total_reward = 0
        done = False
        positions_trace = []
        while not done:
            action = agent.choose_action_deploy(state)
            next_state, reward, done = env.step(action)
            positions_trace.append(env.taxi_position)
            visualize_episode_with_imgs(ax, city, positions, env, positions_trace)
            plt.pause(0.5)
            state = next_state


if __name__ == "__main__":
    main()
