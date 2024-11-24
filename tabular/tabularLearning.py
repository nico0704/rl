import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import time

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

# weights = {
#     (1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 1): 1, (2, 4): 2, (2, 5): 4,
#     (3, 6): 5, (5, 6): 6, (1, 7): 3, (7, 8): 1, (8, 5): 4, (6, 9): 3,
#     (9, 10): 1, (10, 4): 2, (7, 11): 4, (11, 12): 3, (12, 8): 2,
#     (9, 13): 3, (13, 14): 4, (14, 10): 1
# }

class TaxiEnvironment:
    def __init__(self, city_graph):
        self.city = city_graph
        self.reset()

    def reset(self):
        self.taxi_position = random.choice(list(self.city.nodes))
        self.passenger_status = 'no_passenger'
        return (self.taxi_position, self.passenger_status)

    def step(self, action):
        if action in self.city[self.taxi_position]:
            edge_weight = self.city[self.taxi_position][action]['weight']
            self.taxi_position = action
        else:
            return (self.taxi_position, self.passenger_status), -10, False  # Invalid action

        reward = -edge_weight  # Movement costs energy proportional to weight
        done = False

        if self.passenger_status == 'no_passenger' and self.taxi_position == self.passenger_start:
            self.passenger_status = 'has_passenger'
            reward += 10  # Reward for picking up passenger
        elif self.passenger_status == 'has_passenger' and self.taxi_position == self.passenger_destination:
            reward += 20  # Reward for successful delivery
            done = True

        return (self.taxi_position, self.passenger_status), reward, done

### Q-Learning ###
class QLearningAgent:
    def __init__(self, env, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Lernrate
        self.gamma = gamma  # Diskontierungsfaktor
        self.epsilon = epsilon  # Epsilon-Greedy-Strategie
        self.q_table = {}

        # Q-Tabelle initialisieren
        for node in env.city.nodes:
            for status in ['no_passenger', 'has_passenger']:
                state = (node, status)
                self.q_table[state] = {neighbor: 0 for neighbor in env.city[node]}

    def choose_action(self, state):
        # Epsilon-Greedy-Strategie
        if random.random() < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        # Q-Value-Update-Regel
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

def visualize_episode_with_passenger_information(env, agent, city, positions):
    state = env.reset()
    env.taxi_position = 12 # set taxi start position manually
    positions_trace = [env.taxi_position]
    passenger_start = env.passenger_start
    passenger_destination = env.passenger_destination

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        positions_trace.append(env.taxi_position)
        state = next_state
    
    print(f"path: {positions_trace}")

    fig, ax = plt.subplots(figsize=(8, 8))
    def update(num):
        ax.clear()        
        nx.draw(city, pos=positions, with_labels=True, node_color='lightblue', node_size=800, ax=ax)        
        nx.draw_networkx_nodes(city, pos=positions, nodelist=[passenger_start], node_color='green', node_size=1000, ax=ax)        
        nx.draw_networkx_nodes(city, pos=positions, nodelist=[passenger_destination], node_color='red', node_size=1000, ax=ax)        
        nx.draw_networkx_nodes(city, pos=positions, nodelist=[positions_trace[num]], node_color='yellow', node_size=1000, ax=ax)
        edge_labels = nx.get_edge_attributes(city, 'weight')
        nx.draw_networkx_edge_labels(city, pos=positions, edge_labels=edge_labels)
        # legend for colors
        labels = ["Passanger-Start", "Passenger-Destination", "Taxi"]
        colors = ["green", "red", "yellow"]
        for color in colors:
            plt.scatter([], [], color=color, label=labels[colors.index(color)])
        plt.legend(loc="upper left")
    
    time.sleep(1)
    ani = animation.FuncAnimation(fig, update, frames=len(positions_trace), interval=1000, repeat=False)
    plt.show()
    
def save_q_table(agent):
    with open('qtable.txt', 'w') as file:
        for entry in agent.q_table:
            formatted_values = {action: f"{value:.2f}" for action, value in agent.q_table[entry].items()}
            # Schreibe den State und die formatierten Q-Werte in die Datei
            file.write(f"{entry}: {formatted_values}\n")


### define city graph ###
city = nx.Graph()
city.add_edges_from(edges)
# Assign random weights to each edge
for u, v in city.edges():
    city[u][v]['weight'] = random.randint(1, 5)
# nx.set_edge_attributes(city, weights, 'weight')
edge_labels = nx.get_edge_attributes(city, 'weight')
plt.figure(figsize=(8, 8))
nx.draw(city, pos=positions, with_labels=True, node_color='lightblue', node_size=800)
nx.draw_networkx_edge_labels(city, pos=positions, edge_labels=edge_labels)
plt.title("Stadt-Graph")
plt.show()

### training ###
env = TaxiEnvironment(city)
agent = QLearningAgent(env)
start = 10
destination = 8
env.passenger_start = start
env.passenger_destination = destination

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward

save_q_table(agent)
visualize_episode_with_passenger_information(env, agent, city, positions)

