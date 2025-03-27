import config
import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(path):
    config_dict = {
        "run_name": config.run_name,
        "max_ep": config.max_ep,
        "max_train_timestemps": config.max_train_timestemps,
        "action_std": config.action_std,
        "action_std_decay_rate": config.action_std_decay_rate,
        "min_action_std": config.min_action_std,
        "action_std_decay_freq": config.action_std_decay_freq,
        "update_timestep": config.update_timestep,
        "K_epochs": config.K_epochs,
        "eps_clip": config.eps_clip,
        "gamma": config.gamma,
        "lr_actor": config.lr_actor,
        "lr_critic": config.lr_critic,
        "render_freq": config.render_freq,
        "print_freq": config.print_freq,
        "save_model_freq": config.save_model_freq,
        "render": config.render,
        "screen_width": config.screen_width,
        "screen_height": config.screen_height,
        "sensor_dim": config.sensor_dim,
        "num_checkpoints": config.num_checkpoints,
        "track_width": config.track_width,
        "track_radius": config.track_radius
    }
    with open(path, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    print(f"Config saved to {path}")
    
    
def get_latest_run_name(runs_dir="runs"):
    run_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_folders:
        raise FileNotFoundError("No runs found in the 'runs' directory.")
    run_folders.sort(key=lambda name: os.path.getmtime(os.path.join(runs_dir, name)))
    return run_folders[-1]  # latest