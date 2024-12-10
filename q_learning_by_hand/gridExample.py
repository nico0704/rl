import numpy as np

# Define grid enviorment
num_rows, num_cols = 3, 3  
yellow_cell = (0, 0) # Yellow cell
gray_cells = [(1, 1), (1, 2)]  # Gray cell 

actions = ["up", "down", "left", "right", "stay"]  # Possible actions
num_actions = len(actions)
gamma = 1.0  # Discount factor (deterministic)

# Reward and costs
move_cost = 1
stay_cost_normal = 2
stay_cost_yellow = 0
move_cost_gray = 10

# Check for valid move
def is_valid_position(row, col):
    return 0 <= row < num_rows and 0 <= col < num_cols

# Get the next state and reward for a given action
def get_next_state_and_reward(row, col, action):
    if action == "up":
        next_row, next_col = row - 1, col
    elif action == "down":
        next_row, next_col = row + 1, col
    elif action == "left":
        next_row, next_col = row, col - 1
    elif action == "right":
        next_row, next_col = row, col + 1
    elif action == "stay":
        if (row, col) == yellow_cell:
            return row, col, -stay_cost_yellow
        else:
            return row, col, -stay_cost_normal

    # Check for boundaries
    if is_valid_position(next_row, next_col):
        if (next_row, next_col) in gray_cells:
            return next_row, next_col, -move_cost_gray
        else:
            return next_row, next_col, -move_cost
    else:
        return None  # Invalid action

# Initialize the Q-table
Q_table = np.zeros((num_rows, num_cols, num_actions))
iterations = 5

# Value iteration algorithm
for iteration in range(iterations):
    new_Q_table = np.copy(Q_table)
    for row in range(num_rows):
        for col in range(num_cols):
            for action_idx, action in enumerate(actions):
                result = get_next_state_and_reward(row, col, action)
                if result is None:  
                    new_Q_table[row, col, action_idx] = -np.inf  # Mark as invalid action
                else:
                    next_row, next_col, reward = result
                    best_next_Q = np.max(Q_table[next_row, next_col]) # Get best next action
                    new_Q_table[row, col, action_idx] = reward + gamma * best_next_Q # Update Q-table
            Q_table = new_Q_table

    # Print Q-table every iteration
    print(f"Iteration {iteration + 1}:")
    for row in range(num_rows):
        for col in range(num_cols):
            print(f"Q[{row}, {col}] = {Q_table[row, col]}")
        print()

# Print final Q-table
print("Final Q-table:")
for row in range(num_rows):
    for col in range(num_cols):
        print(f"Q[{row}, {col}] = {Q_table[row, col]}")
    print()
