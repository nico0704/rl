#### Environment the PPO-Algorithm is using for training and testing ####
# 
# Usage:
#     via train.py or test.py
# Description:
#     - Creates the environment including different tracks with checkpoints, the car and the renderer
#     - Uses the action from the agent (throttle and steer) to move the car on the track
#     - Calculates the reward based on the actions the agent is taking
#     - Returns the state [car speed, car angle, sensor data] to the agent
#     - After completing or failing a track, a new track is created 

import numpy as np
from env.track import Track
from env.car import Car
from env.renderer import Renderer
import env.utils as Utils

class RaceCarEnv:
    
    def __init__(self, width=1920, height=1080, render=True, sensor_dim=5, num_checkpoints=80, track_width=80, track_radius=300):

        # screen settings
        self.WIDTH, self.HEIGHT = width, height

        # track
        self.track_width = track_width
        self.track_radius = track_radius

        # checkpoints
        self.num_checkpoints = num_checkpoints
        self.current_checkpoint = 0  # start at the first checkpoint
        self.checkpoints_passed = [False] * (self.num_checkpoints + 1)  # extra for finish line

        # sensor_data
        self.sensor_dim = sensor_dim

        self.renderer = Renderer(self.WIDTH, self.HEIGHT, None, None, self.checkpoints_passed) if render else None

        # track, car and renderer
        self.track = None
        self.car = None
        self.reset()


    def reset(self):

        self.current_checkpoint = 0 
        self.checkpoints_passed = [False] * (self.num_checkpoints + 1) 

        self.track = Track(self.WIDTH, self.HEIGHT, self.track_width, self.track_radius, self.num_checkpoints, num_points=1000)
        self.car = Car(self.track, car_radius=5, car_speed=0, MAX_SPEED=5, ACCELERATION=0.2, FRICTION=0.5, TURN_SPEED=3, sensor_count=self.sensor_dim, sensor_range=100)

        if self.renderer is not None:
            self.renderer.car = self.car
            self.renderer.track = self.track
            self.renderer.checkpoints_passed = self.checkpoints_passed

        return self.get_state()
    
    
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


    def check_checkpoint_crossed(self):

        car_x, car_y = self.car.car_position
        car_radius = self.car.car_radius
        checkpoints = self.track.checkpoints

        if self.current_checkpoint < len(checkpoints):
            start, end = checkpoints[self.current_checkpoint]

            # check if the car is close enough to the checkpoint line
            if Utils.point_line_distance((car_x, car_y), start, end) < car_radius:
                self.checkpoints_passed[self.current_checkpoint] = True
                self.current_checkpoint += 1
                return self.current_checkpoint, False  # reward for crossing a checkpoint

        # if all checkpoints are passed, activate finish line
        if self.current_checkpoint == self.num_checkpoints and not self.checkpoints_passed[-1]:
            start, end = checkpoints[-1]  # finish line is at the last checkpoint
            if Utils.point_line_distance((car_x, car_y), start, end) < car_radius * 2:
                self.checkpoints_passed[-1] = True
                return 100.0, True  # reward for completing the track
                
        return 0.0, False # no reward if no checkpoint was crossed
    

    def is_on_track(self):
    
        car_x, car_y = self.car.car_position

        # compute distances to left and right boundary
        distances_to_left = np.hypot(car_x - self.track.left_x, car_y - self.track.left_y)
        distances_to_right = np.hypot(car_x - self.track.right_x, car_y - self.track.right_y)

        # find the closest points on each boundary
        min_dist_left = np.min(distances_to_left)
        min_dist_right = np.min(distances_to_right)

        # ensure car stays within the track including car radius
        return (min_dist_left > self.car.car_radius) and (min_dist_right > self.car.car_radius)
    
    
    def render(self):
        if self.renderer is not None:
            self.renderer.render()

    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def get_dims(self):
        return 2 + len(self.car.sensor_data), 2








