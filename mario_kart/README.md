# Mini-Project Policy Gradient - Mario Kart
This project is the second mini-project for the reinforcement learning course. It simulates a car driving a race track. The cars goal is to drive the track as fast as possible without leaving the track. To train the agent we used the PPO. 

---

### Setup
```bash
pip install numpy
pip install torch torchvision torchaudio
pip install pygame
pip install scipy
pip install pyyaml
```

---

### Environment
The environment folder has the following structure:

env/  
├── [car.py](env/car.py)  
├── [environment.py](env/environment.py)  
├── [renderer.py](env/renderer.py)  
├── [track.py](env/track.py)  
├── [utils.py](env/utils.py)  

A detailed description for each of these files is written as comments on top of the code.

The following code snippet from the [environment.py](env/environment.py#L61) shows what a step based on an action containing throttle and steer looks like.
 During each step the car moves and the reward is calculated based on the crossed checkpoints and the current speed. Also the function checks if the car left or finished the track. After that the function returns the state containing the car speed, the car rotation and the distance of the sensors to the boundaries, the reward and the status.

```python
    def step(self, action):

        self.car.step(action)

        if self.renderer is not None:
            self.renderer.step(action)

        checkpoint_reward, finished = self.check_checkpoint_crossed()
        checkpoint_reward /= self.num_checkpoints

        if self.car.car_speed > 0:
            speed_reward = (self.car.car_speed / self.car.MAX_SPEED) * 0.75
        elif self.car.car_speed < 0:
            speed_reward = (self.car.car_speed / self.car.MAX_SPEED) * 0.2 
        else:
            speed_reward = -0.5 

        off_track = not self.is_on_track()
        track_penalty = -1.0 if off_track else 0.0 

        reward = checkpoint_reward + speed_reward + track_penalty

        return self.get_state(), reward, finished or off_track
    

    def get_state(self):
        # returns status
        return np.hstack(([self.car.car_speed], [self.car.car_angle / 180], self.car.sensor_data))
```

---

### Configuration:
The following parameters can be changed in the [config.py](/mario_kart/config.py):

###### Training and testing parameter:
- `run_name`: name of the current run | default: ppo_{data_str}
- `max_ep`: maximal number of timesteps in one episode | default: 1000
- `max_training_timesteps`: maximal timesteps for training loop | default: int(3e6) 

- `action_std`: sets the initial standard deviation of the action distribution in a continuous action space | default: 0.6
- `action_std_decay_rate`: controls how quickly the exploration decreases over time | default: 0.5
- `min_action_std`: minimal value for standard deviation of the action distribution | default: 0.1
- `action_std_decay_freq`: defines how often the action_std should be reduced (in number of timesteps)  | default: int(2.5e5)

- `update_timestep`: every n timesteps update policy | default: max_ep * 4
- `K_epoches`: one PPO update contains a policy update for K epochs | default: 80
- `eps_clip`: limits how much new policy can deviate from the old one during updates | default: 0.2
- `gamma`: determines how much future rewards are worth compared to immediate rewards | default: 0.99
- `lr_actor`: learning rate for actor network | default: 0.0003
- `lr_critic`: learning rate for critic network | default: 0.001

###### Render Parameter

- `render_freq`: interval for rendering | default: 50
- `print_freq`: interval for printing the average reward (in number of timesteps) | default: max_ep * 2
- `save_model_freq`: interval for saving model (in number of timesteps) | default: int(1e5) 
- `render`: boolean for rendering | default: True
- `screen_wdith`: screen width of the pygame window | default: 1000
- `screen_height`: screen height of the pygame window | default: 700

###### Environment Parameter

- `sensor_dim`: number of sensors the car is using equally distributed between -90 and 90 degrees | default: 5
- `num_checkpoints`: number of checkpoints on the track | default: 80
- `track_width`: width of the track | default: 80
- `track_radius`: radius of the circuit track | default: 300

---
### [Train](/mario_kart/train.py)
This section describes the training-part for the driving simulation. A more detailed description is written in the [train.py](/mario_kart/train.py#L0) as a comment on top of the code. 

#### Run:
```bash
python train.py
```

---

### PPO Algorithm
We decided to use the PPO Algorithm because it is good for continuous actions like throttling and steering. Here is some short explaination of the PPO algorithm in our project.

##### Basics
PPO (Proximal Policy Optimization) is a widely used policy-gradient method in reinforcement learning. It updates the policy in a stable and sample-efficient way by preventing too large policy updates using a clipped surrogate objective.
This implementation is based on an Actor-Critic architecture with a stochastic Gaussian policy, where actions are sampled from a learned normal distribution.

##### Main Components
1. Actor-Critic Network
The policy consists of two neural networks:
    - Actor: Outputs the mean of the action distribution
    - Critic: Estimates the value of the current state

    ```python 
    # actor
    self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Tanh()
                )
    # critic
    self.critic = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
    ```
    [ppo.py line 40-56](ppo.py#L40)

2. Stochastic Action Sampling
The actor produces the mean of a Gaussian distribution, and the action is sampled from it using a covariance matrix.

    ```python
    action_mean = self.actor(state)
    cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    dist = MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()
    ```
    [ppo.py line 62-65](ppo.py#L62)

3. Decaying Action Standard Deviation
The standard deviation of the action distribution is gradually decreased during training to reduce exploration over time and encourage exploitation of learned behavior.

    ```python
    def decay_action_std(self, decay_rate, min_std):
        self.action_std = max(self.action_std - decay_rate, min_std)
        self.set_action_std(self.action_std)
    ```
    [ppo.py line 103-105](ppo.py#L103)

4. Policy Update using Clipped Objective
PPO improves the policy by maximizing the clipped objective function. This avoids large updates that could destabilize learning.

    ```python
    # Finding the ratio (pi_theta / pi_theta__old)
    ratios = torch.exp(logprobs - old_logprobs.detach())

    # Finding Surrogate Loss  
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    ```
    [ppo.py line 144-149](ppo.py#L144)

---

### [Test](/mario_kart/test.py)
This section describs the testing-part for the mario kart game. A more detailed description of the file is [here](/mario_kart/test.py#L0). 

#### Run:
```bash
python test.py --run_name <run-folder> --num_episodes <num-testing-episodes>
```
- `--run_name`: name of the run folder to load (if omitted latest is used)
- `--num_episodes`: number of episodes to run for testing (default: 10)

---

### Results
<h4 align="center"><b>The agent can perform well on various complex that have different levels of complexity</b></h4>
<p align="center">
  <img src="results/track1.gif" width="30%" />
  <img src="results/track2.gif" width="30%" />
  <img src="results/track3.gif" width="30%" />
</p>
<br>
<br>
<br>
<h4 align="center"><b>Demo of training process</b></h4>
<p align="center">
  <a href="https://youtu.be/Y9vWKYbkDhk">
    <img src="https://img.youtube.com/vi/Y9vWKYbkDhk/0.jpg" alt="Demo-Video">
  </a>
</p>



