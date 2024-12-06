# Reinforcment Learning Course WISE 24/25

This repository is a collection of solutions for the reinforcement learning course of Group Three.


## Pre-Project (three-wheel-robot)
This project simulates the movement of a three wheel robot form a starting-point to an ending-point

### Usage
How to run this:
```bash
cd three_wheel_robot
python twr.py
```

#### Results
These are three example-results of the pre-project:
<div>
    <img src="/three_wheel_robot/visus/1.png" alt="First Result" width="350">
    <img src="/three_wheel_robot/visus/2.png" alt="First Result" width="350">
    <img src="/three_wheel_robot/visus/3.png" alt="First Result" width="350">
</div>


## First Mini-Project (tabular)
This project is the first mini-project for the reinforcement learning course. It simulates a taxi-driving-system. The taxi drivers goal is to get a passenger and deliver him/her to his/hers destination. It's realised as a graph problem, so the goal is to find the shortest path. To find the shortest path we used Q-Learning. 

### Train

#### Configuration:
- `EPISODES`: The number of episodes to train the agent. Default: 100000
- `VISU_EPISODE`: The interval at which training progress is visualized (if visualization is enabled). Default: 1000
- `SAVE_Q_TABLE`: Whether to save the Q-Table after training. Default: True
- `SAVE_Q_TABLE_PATH`: The path to save the Q-Table. Default: q_tables/qt_{EPISODES}.txt
- `VISUALIZE_TRAINING`: Enable/disable visualization of the training process. Default: False
- `PLOT`: Enable/disable plotting of training metrics. Default: False

```bash
cd tabular
python train_crazy_taxi.py
```

### Run
#### Configuration:
- `Q_TABLE_PATH`: The path to the pre-trained Q-Table used for running the simulation. Default: q_tables/qt_10000000.txt
- `ITERATIONS`: The number of iterations the simulation will run. Default: 10

```bash
cd tabular
python run_crazy_taxi.py
```

#### Results
This is one example result for the first mini-project
<div>
    <img src="/tabular/plots/cumulative_reward_plot.png" alt="First Result" width="350">
    <img src="/tabular/plots/total_reward_plot.png" alt="First Result" width="350">
</div>
To see some more results run the code. There are some more visualizations included.