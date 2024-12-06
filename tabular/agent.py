import random


#### agent ####
class QLearningAgent:
    def __init__(self, env, mode="train", alpha=0.01, gamma=0.9, epsilon=0.1, q_table_path=None):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.mode = mode
        self.q_table_path = q_table_path
        if self.mode == "train":
            self.q_table = self._initialize_q_table()
        if self.mode == "run":
            self.q_table = self.load_q_table()

    def _initialize_q_table(self):
        q_table = {}
        for node in self.env.city.nodes:
            for start in self.env.city.nodes:
                for dest in self.env.city.nodes:
                    for status in ['no_passenger', 'has_passenger']:
                        state = (node, status, start, dest)
                        q_table[state] = {neighbor: 0 for neighbor in self.env.city[node]}
        return q_table

    def choose_action(self, state):
        if random.random() < self.epsilon and self.mode == "train":
            # Epsilon-Greedy Strategy
            return random.choice(list(self.q_table[state].keys()))
        return max(self.q_table[state], key=self.q_table[state].get)

    def choose_action_deploy(self, state):
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_future_q - self.q_table[state][action]
        )
        
    def save_q_table(self, path):
        with open(path, "w") as file:
            for entry in self.q_table:
                formatted_values = {
                    action: f"{value:.2f}"
                    for action, value in self.q_table[entry].items()
                }
                file.write(f"{entry}: {formatted_values}\n")

    def load_q_table(self):
        q_table = {}
        with open(self.q_table_path, "r") as file:
            for line in file:
                state_str, actions_str = line.strip().split(": ", 1)
                state = eval(state_str)                
                actions = eval(actions_str)
                actions = {int(action): float(value) for action, value in actions.items()}
                q_table[state] = actions
        return q_table

