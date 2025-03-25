from datetime import datetime

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

render = True
render_freq = 50
print_freq = max_ep * 2
save_model_freq = int(1e4)