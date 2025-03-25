# Mini-Project Policy Gradient - Mario Kart
This project is the second mini-project for the reinforcement learning course. It simulates the mario kart game. The cars goal is to drive the track as fast as possible without leaving the track. To train the agent we used the PPO. 

---

### Setup
```bash
pip install torch ???
pip install pygame
```

---

### [Train](/mario_kart/train_ppo.py)
This section describs the training-part for the mario kart game. A more detailed description of the file is [here](/mario_kart/README.md#project-description). 

#### Configuration:
##### Environment hyperparameter
- `has_continuous_action_space`: defines if the action space is continuous (False: discrete) | default: True
- `max_ep_len`: maximal number of timesteps in one episode | default: 1000
- `max_training_timesteps`: maximal timesteps for training loop | default: int(3e6) 
- `print_freq`: interval for printing the average reward (in number of timesteps) | default: max_ep_len * 2
- `log_freq`: interval for logging the average reward (in number of timesteps) | default: max_ep_len * 2
- `save_model_freq`: interval for saving model (in number of timesteps) | default: int(1e5) 
- `action_std`: sets the initial standard deviation of the action distribution in a continuous action space | default: 0.6
- `action_std_decay_rate`: controls how quickly the exploration decreases over time | default: 0.5
- `min_action_std`: minimal value for standard deviation of the action distribution | default: 0.1
- `action_std_decay_freq`: defines how often the action_std should be reduced (in number of timesteps)  | default: int(2.5e5)

##### PPO hyperparameter
- `update_timestep`: every n timesteps update policy | default: max_ep_len * 4
- `K_epoches`: one PPO update contains a policy update for K epochs | default: 80
- `eps_clip`: limits how much new policy can deviate from the old one during updates | default: 0.2
- `gamma`: determines how much future rewards are worth compared to immediate rewards | default: 0.99
- `lr_actor`: learning rate for actor network | default: 0.0003
- `lr_critic`: learning rate for critic network | default: 0.001
- `random_seed`: sets the starting point for random number generation (0 = no random seed) | default: 0 (!kann eigentlich raus weil wird so nicht benutzt)

#### Run:
```bash
python train_ppo.py
```

---

### [Test](/mario_kart/test.py)
This section describs the testing-part for the mario kart game. A more detailed description of the file is [here](/mario_kart/README.md#project-description). 

#### Configuration:
- `render`: defines whether the track should be rendered | default: True
- `total_test_episodes`: defines the number of test episodes | default: 10

#### Run:
```bash
python test.py
```

---

### Results

--- 

### Project-Description

#### [PPO](ppo.py)

#### [Environment](race_car_env.py)

#### [Training](train_ppo.py)

#### [Testing](test.py)