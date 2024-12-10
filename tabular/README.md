# Mini-Project Tabular - Crazy Taxi
This project is the first mini-project for the reinforcement learning course. It simulates a taxi-driving-system. The taxi drivers goal is to get a passenger and deliver him/her to his/hers destination. It's realised as a graph problem, so the goal is to find the shortest path. To find the shortest path we used Q-Learning. 

### Setup
```bash
pip install -r requirements.txt
```

### Train

#### Configuration:
- `EPISODES`: The number of episodes to train the agent. Default: 100000
- `VISU_EPISODE`: The interval at which training progress is visualized (if visualization is enabled). Default: EPISODES / 10
- `SAVE_Q_TABLE`: Whether to save the Q-Table after training. Default: True
- `SAVE_Q_TABLE_PATH`: The path to save the Q-Table. Default: q_tables/qt_{EPISODES}.txt
- `VISUALIZE_TRAINING`: Enable/disable visualization of the training process. Default: False
- `PLOT`: Enable/disable plotting of training metrics. Default: False

```bash
python train_crazy_taxi.py
```

### Run
#### Configuration:
- `Q_TABLE_PATH`: The path to the pre-trained Q-Table used for running the simulation. Default: q_tables/qt_100000.txt
- `ITERATIONS`: The number of iterations the simulation will run. Default: 10

```bash
python run_crazy_taxi.py
```

#### Results
![crazy taxi](assets/visu.gif)

This is one example result for the first mini-project
<div>
    <img src="/tabular/plots/cumulative_reward_plot.png" alt="First Result" width="500">
    <img src="/tabular/plots/total_reward_plot.png" alt="First Result" width="500">
</div>
To see some more results run the code. There are some more visualizations included.

#### Mathematical Concept

`Q(s,a)= r(s,a) + γ 𝔼[V*(Sₜ)]` 

```python
def update_q_value(self, state, action, reward, next_state):
    max_future_q = max(self.q_table[next_state].values())
    self.q_table[state][action] += self.alpha * (
        reward + self.gamma * max_future_q - self.q_table[state][action]
)
```

- `Q(s,a)` → `self.q_table[state][action]` is the q-table-value that gets updated 
- `r(s,a) ` → ` reward ` is the reward based on the action of the current state
- ` γ ` → ` self.gamma ` is the discount-factor 
- ` V*(Sₜ) ` → `max_future_q` is the maximal value from the q-table based on next-state 
- `𝔼` is in our case always 1 because of our deterministic environment  

`self.alpha` is the learning-rate that ensures that the policy is slowly converging to its optimal value

The subtraction of `self.q_table[state][action]` measures how the current value and the new value differ from each other. It is called the Temporal Difference Error (TDE). The TDE determines the extent to which the Q-table value is adjusted.