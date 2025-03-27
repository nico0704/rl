from datetime import datetime

# Training and Testing
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"ppo_{date_str}"

max_ep = 1000
max_train_timestemps = int(3e6)

action_std = 0.6
action_std_decay_rate = 0.05
min_action_std = 0.1
action_std_decay_freq = int(2.5e5)

update_timestep = max_ep * 4
K_epochs = 80
eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001

# Renderer and Logging
render = True
render_freq = 50
print_freq = max_ep * 4
save_model_freq = int(1e4)
screen_width = 1000
screen_height = 700

# Environment
sensor_dim = 5
num_checkpoints = 80
track_width = 80
track_radius = 300