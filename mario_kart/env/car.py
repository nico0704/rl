#### The car which is driving on the tracks ####
# 
# Usage:
#     via environment.py
# Description:
#     - Stores the current speed and the radius of the car
#     - Stores the car limits given by the environment like maximal speed, acceleration, turn speed or friction
#     - Calculates the distance between the car and the boundaries of the track by using the sensors of the car
#     - Performs a part of the action by moving the car with the step function

import numpy as np
import env.utils as Utils

class Car:

    def __init__(self, track, car_radius, car_speed, MAX_SPEED, ACCELERATION, FRICTION, TURN_SPEED, sensor_count, sensor_range):

        # track
        self.track = track

        # car
        center_x = self.track.center_x
        center_y = self.track.center_y
        self.car_position = np.array([center_x[0], center_y[0]], dtype=np.float32)
        dx = center_x[0] - center_x[1]
        dy = center_y[0] - center_y[1]
        self.car_angle = np.degrees(np.arctan2(dy, dx))
        self.car_radius = car_radius
        self.car_speed = car_speed
        self.MAX_SPEED = MAX_SPEED
        self.ACCELERATION = ACCELERATION
        self.FRICTION = FRICTION
        self.TURN_SPEED = TURN_SPEED

        # sensors
        self.sensor_angles = np.linspace(-90, 90, sensor_count, endpoint=True)
        self.sensor_range = sensor_range
        self.sensor_data = self.get_sensor_readings()

    
    def get_sensor_readings(self):

        readings = []
        car_x, car_y = self.car_position

        for angle_offset in self.sensor_angles:
            sensor_angle = np.radians(self.car_angle) + np.radians(angle_offset)
            sensor_dx = np.cos(sensor_angle)
            sensor_dy = np.sin(sensor_angle)

            # define the sensor line
            sensor_end_x = car_x + sensor_dx * self.sensor_range
            sensor_end_y = car_y + sensor_dy * self.sensor_range
            sensor_start = (car_x, car_y)
            sensor_end = (sensor_end_x, sensor_end_y)

            min_dist = self.sensor_range  # default max sensor range

            # query nearby indices from KD-Trees
            left_nearest_idx = sorted(self.track.left_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range))
            right_nearest_idx = sorted(self.track.right_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range))

            for i in range(len(left_nearest_idx) - 1):
                idx1 = left_nearest_idx[i]
                idx2 = left_nearest_idx[i + 1]
                boundary_start = (self.track.left_x[idx1], self.track.left_y[idx1])
                boundary_end = (self.track.left_x[idx2], self.track.left_y[idx2])

                intersection = Utils.line_intersection(sensor_start, sensor_end, boundary_start, boundary_end)
                if intersection:
                    dist = np.hypot(intersection[0] - car_x, intersection[1] - car_y)
                    if dist < min_dist:
                        min_dist = dist

            for i in range(len(right_nearest_idx) - 1):
                idx1 = right_nearest_idx[i]
                idx2 = right_nearest_idx[i + 1]
                boundary_start = (self.track.right_x[idx1], self.track.right_y[idx1])
                boundary_end = (self.track.right_x[idx2], self.track.right_y[idx2])

                intersection = Utils.line_intersection(sensor_start, sensor_end, boundary_start, boundary_end)
                if intersection:
                    dist = np.hypot(intersection[0] - car_x, intersection[1] - car_y)
                    if dist < min_dist:
                        min_dist = dist

            readings.append(min_dist)

        return np.array(readings, dtype=np.float32)
    

    def step(self, action):
        throttle, steer = action  # [throttle (-1 to 1), steel (-1 to 1)]

        # update speed
        self.car_speed += throttle * self.ACCELERATION
        self.car_speed = np.clip(self.car_speed, -self.MAX_SPEED, self.MAX_SPEED)

        # update direction
        self.car_angle += steer * self.TURN_SPEED
        self.car_angle = (self.car_angle + 180) % 360 - 180

        # move car
        dx = np.cos(np.radians(self.car_angle)) * self.car_speed
        dy = np.sin(np.radians(self.car_angle)) * self.car_speed
        self.car_position += np.array([dx, dy])

        # calculate sensor data
        self.sensor_data = self.get_sensor_readings()

        # return car speed
        return np.hstack(([self.car_speed], [self.car_angle / 180], self.sensor_data))

        