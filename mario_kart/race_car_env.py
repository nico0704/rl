import pygame
import numpy as np
import matplotlib.pyplot as plt

class RaceCarEnv:

    def __init__(self, track_file, track_width=40, track_radius=300, render_mode="human"):
        # screen settings
        self.WIDTH, self.HEIGHT = 1000, 1000
        self.track_width = track_width
        self.track_radius = track_radius

        # generate track
        self.center_x, self.center_y = self.generate_wavy_loop(num_points=200)
        self.left_x, self.left_y, self.right_x, self.right_y = self.compute_boundaries()

        # car
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32) #start position
        self.car_angle = 0
        self.car_speed = 0
        self.MAX_SPEED = 5
        self.ACCELERATION = 0.2 #beschleunigung
        self.FRICTION = 0.05 #reibung
        self.TURN_SPEED = 3 

        # rendering
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()

    def generate_circle_track(self, num_points=100):
        """ Generates centerline points for a circular track. """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        center_x = self.WIDTH // 2 + self.track_radius * np.cos(angles)
        center_y = self.HEIGHT // 2 + self.track_radius * np.sin(angles)
        return center_x, center_y

    def generate_wavy_loop(self, num_points=150, amplitude=50, frequency=3):
        """Generates a closed-loop wavy track."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        center_x = self.WIDTH // 2 + (self.track_radius + amplitude * np.sin(frequency * angles)) * np.cos(angles)
        center_y = self.HEIGHT // 2 + (self.track_radius + amplitude * np.sin(frequency * angles)) * np.sin(angles)
        return center_x, center_y

    def compute_boundaries(self):
        """ Computes left and right track boundaries based on the centerline. """
        left_x, left_y = [], []
        right_x, right_y = [], []

        num_points = len(self.center_x)

        for i in range(num_points):
            # Compute tangent (direction)
            dx = self.center_x[(i + 1) % num_points] - self.center_x[i]
            dy = self.center_y[(i + 1) % num_points] - self.center_y[i]

            # Normalize tangent
            length = np.hypot(dx, dy)
            dx /= length
            dy /= length

            # Compute perpendicular vector
            perp_x = -dy
            perp_y = dx

            # Offset points to get left and right boundaries
            half_width = self.track_width / 2
            left_x.append(self.center_x[i] + half_width * perp_x)
            left_y.append(self.center_y[i] + half_width * perp_y)
            right_x.append(self.center_x[i] - half_width * perp_x)
            right_y.append(self.center_y[i] - half_width * perp_y)

        return np.array(left_x), np.array(left_y), np.array(right_x), np.array(right_y)

    # generates left and right boundaries based on the centerline
    def generate_track_boundaries(self, center_x, center_y, track_width):
        left_x, left_y = [], []
        right_x, right_y = [], []
        
        num_points = len(center_x)

        for i in range(num_points):
            # Compute tangent vector (using next point, or previous for last point)
            if i < num_points - 1:
                dx = center_x[i + 1] - center_x[i]
                dy = center_y[i + 1] - center_y[i]
            else:
                dx = center_x[i] - center_x[i - 1]
                dy = center_y[i] - center_y[i - 1]

            # Normalize tangent vector
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            dx /= length
            dy /= length

            # Compute normal (perpendicular) vector
            perp_x = -dy
            perp_y = dx

            # Move to left and right using fixed track width
            half_width = track_width / 2
            left_x.append(center_x[i] + half_width * perp_x)
            left_y.append(center_y[i] + half_width * perp_y)

            right_x.append(center_x[i] - half_width * perp_x)
            right_y.append(center_y[i] - half_width * perp_y)

        return np.array(left_x), np.array(left_y), np.array(right_x), np.array(right_y)

    def step(self, action):
        throttle, steer = action # [throttle (-1 to 1), steering (-1 to 1)] (throttle = drosseln | steering = lenken)

        # update speed
        self.car_speed += throttle * self.ACCELERATION
        self.car_speed = np.clip(self.car_speed, -self.MAX_SPEED, self.MAX_SPEED)

        # update orientation
        self.car_angle += steer * self.TURN_SPEED

        # move car
        dx = np.cos(np.radians(self.car_angle)) * self.car_speed
        dy = np.sin(np.radians(self.car_angle)) * self.car_speed
        self.car_position += np.array([dx, dy])

        # collision check 
        done = not self.is_on_track()
        reward = -10 if done else -0.01 #penalty for leaving the track

        # return state, reward and done flag
        return np.hstack(([self.car_speed], [self.car_angle / 180])), reward, done
    
    # checks if car is inside the track boundaries
    def is_on_track(self):
        car_x, car_y = self.car_position  

        # Compute distances to left and right boundary
        distances_to_left = np.hypot(car_x - self.left_x, car_y - self.left_y)
        distances_to_right = np.hypot(car_x - self.right_x, car_y - self.right_y)

        # Find the closest points on each boundary
        min_dist_left = np.min(distances_to_left)
        min_dist_right = np.min(distances_to_right)

        # Ensure car stays within the track including car radius
        car_radius = 10  # Adjust this if the car size is different
        return (min_dist_left > car_radius) and (min_dist_right > car_radius)


    # draws environment in pygame
    def render(self):
        if self.render_mode != "human":
            return
        
        self.screen.fill((0, 0, 0)) #green background

        # Draw track boundaries
        pygame.draw.lines(self.screen, (255, 255, 255), closed=True, 
        points=list(zip(self.left_x, self.left_y)), width=2)
        pygame.draw.lines(self.screen, (255, 255, 255), closed=True, 
        points=list(zip(self.right_x, self.right_y)), width=2)


        # Draw car
        pygame.draw.circle(self.screen, (255, 0, 0), self.car_position.astype(int), 10)

        pygame.display.flip()
        self.clock.tick(30)

    def reset(self):
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32)
        self.car_speed = 0
        self.car_angle = 0
        return np.hstack(([self.car_speed], [self.car_angle / 180]))

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
    