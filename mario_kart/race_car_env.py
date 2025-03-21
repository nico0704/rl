import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class RaceCarEnv:

    def __init__(self, track_width=40, track_radius=300, render_mode="human"):
        # screen settings
        self.WIDTH, self.HEIGHT = 1000, 1000
        self.track_width = track_width
        self.track_radius = track_radius

        # generate track
        self.num_points = 1000
        self.center_x, self.center_y = self.generate_wavy_loop()
        self.left_x, self.left_y, self.right_x, self.right_y = self.compute_boundaries()

        # Generate KD-Trees for boundary points
        self.left_boundary_tree = KDTree(np.column_stack((self.left_x, self.left_y)))
        self.right_boundary_tree = KDTree(np.column_stack((self.right_x, self.right_y)))

        # Generate checkpoints
        self.num_checkpoints = 50
        self.checkpoints = self.generate_checkpoints()

        # car
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32) #start position
        self.car_radius = 5
        self.car_angle = -80
        self.car_speed = 0
        self.MAX_SPEED = 5
        self.ACCELERATION = 0.2 #beschleunigung
        self.FRICTION = 0.05 #reibung
        self.TURN_SPEED = 3

        # Load the car image
        self.car_image = pygame.image.load("images/f1_top_view.png") 
        self.car_image = pygame.transform.scale(self.car_image, (30, 30)) 
        
        # sensors
        self.sensor_angles = np.linspace(-90, 90, 5, endpoint=True)
        self.sensor_range = self.track_width / 2
        self.sensor_data = []

        # checkpoint tracking
        self.current_checkpoint = 0  # Start at the first checkpoint
        self.checkpoints_passed = [False] * (self.num_checkpoints + 1)  # Extra for finish line

        # rendering
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()

    def generate_wavy_loop(self, amplitude=50, frequency=3):
        """Generates a closed-loop wavy track."""
        angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        np.append(angles, 0.0)
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

    def generate_checkpoints(self):
        """Generate evenly spaced checkpoints perpendicular to the track."""
        checkpoints = []
        step = len(self.center_x) // self.num_checkpoints

        for i in range(self.num_checkpoints):
            idx = i * step
            dx = self.center_x[(idx + 1) % len(self.center_x)] - self.center_x[idx]
            dy = self.center_y[(idx + 1) % len(self.center_y)] - self.center_y[idx]

            # Normalize
            length = np.hypot(dx, dy)
            dx /= length
            dy /= length

            # Compute perpendicular
            perp_x = -dy
            perp_y = dx

            # Create checkpoint line
            half_width = self.track_width / 2
            checkpoint_start = (self.center_x[idx] + half_width * perp_x, self.center_y[idx] + half_width * perp_y)
            checkpoint_end = (self.center_x[idx] - half_width * perp_x, self.center_y[idx] - half_width * perp_y)

            checkpoints.append((checkpoint_start, checkpoint_end))

        # Move the first checkpoint to the end of the list (finish line)
        return list(reversed(checkpoints))

    def check_checkpoint_crossed(self):
        """Check if the car has crossed the next checkpoint in order."""
        car_x, car_y = self.car_position

        if self.current_checkpoint < len(self.checkpoints):
            start, end = self.checkpoints[self.current_checkpoint]

            # Check if the car is close enough to the checkpoint line
            if self.point_line_distance((car_x, car_y), start, end) < self.car_radius:
                self.checkpoints_passed[self.current_checkpoint] = True
                self.current_checkpoint += 1
                print(f'CHECKPOINT {self.current_checkpoint} CROSSED!')
                return self.current_checkpoint  # Reward for crossing a checkpoint

        # If all checkpoints are passed, activate finish line
        if self.current_checkpoint == self.num_checkpoints and not self.checkpoints_passed[-1]:
            start, end = self.checkpoints[-1]  # Finish line is at the last checkpoint
            if self.point_line_distance((car_x, car_y), start, end) < self.car_radius:
                self.checkpoints_passed[-1] = True
                print('FINISH LINE CROSSED!')
                return 100  # Reward for completing the track
                

        return 0  # No reward if no checkpoint was crossed
    
    def point_line_distance(self, point, line_start, line_end):
        """ Berechnet die minimale Distanz eines Punktes zu einer Linie """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # direction of the line
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq == 0:  # if start and end is identical
            return np.hypot(px - x1, py - y1)

        # project of the point on the line
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        t = max(0, min(1, t))  # limits t

        # nearest point to the line
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        # distance between point and line
        distance = np.hypot(px - nearest_x, py - nearest_y)

        return distance

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
        throttle, steer = action  # [throttle (-1 bis 1), lenken (-1 bis 1)]

        # Geschwindigkeit aktualisieren
        self.car_speed += throttle * self.ACCELERATION
        self.car_speed = np.clip(self.car_speed, -self.MAX_SPEED, self.MAX_SPEED)

        # Richtung aktualisieren
        self.car_angle += steer * self.TURN_SPEED

        # Auto bewegen
        dx = np.cos(np.radians(self.car_angle)) * self.car_speed
        dy = np.sin(np.radians(self.car_angle)) * self.car_speed
        self.car_position += np.array([dx, dy])

        # Sensorwerte aktualisieren
        self.sensor_data = self.get_sensor_readings()

        # **Checkpoints belohnen (Belohnung zwischen 0 und 1 normalisieren)**
        checkpoint_reward = self.check_checkpoint_crossed()
        checkpoint_reward = checkpoint_reward / 100.0  # Maximal 1.0 für Ziellinie

         # **Geschwindigkeitsbelohnung normalisieren**
        if self.car_speed > 0:
            speed_reward = (self.car_speed / self.MAX_SPEED) * 0.8  # Vorwärts max 0.8 Belohnung
        elif self.car_speed < 0:
            speed_reward = (self.car_speed / self.MAX_SPEED) * 0.5  # Rückwärts max -0.3 Strafe
        else:
            speed_reward = -1.0  # Stillstand wird stärker bestraft

        # **Strafe für das Verlassen der Strecke**
        off_track = not self.is_on_track()
        track_penalty = -1.0 if off_track else 0.0  # Harte Strafe für das Verlassen der Strecke

        # **Finaler Reward: Kombiniere alle Belohnungen**
        reward = checkpoint_reward + speed_reward + track_penalty

        # **Falls das Auto von der Strecke ist, Episode beenden**
        done = off_track or checkpoint_reward >= 1.0  # Episode endet, wenn das Auto ins Ziel kommt oder off-track ist

        # **Status zurückgeben**
        return np.hstack(([self.car_speed], [self.car_angle / 180], self.sensor_data)), reward, done

    def line_intersection(self, p1, p2, p3, p4):
        """Returns the intersection point of two line segments (p1, p2) and (p3, p4), or None if no intersection."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Compute determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:  # Parallel lines or too close
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)

        return None
    

    def get_sensor_readings(self):
        """Returns the distances from the car to the nearest track boundary for each sensor."""
        readings = []
        car_x, car_y = self.car_position

        for angle_offset in self.sensor_angles:
            sensor_angle = np.radians(self.car_angle) + np.radians(angle_offset)
            sensor_dx = np.cos(sensor_angle)
            sensor_dy = np.sin(sensor_angle)

            # Define the sensor line
            sensor_end_x = car_x + sensor_dx * self.sensor_range
            sensor_end_y = car_y + sensor_dy * self.sensor_range
            sensor_start = (car_x, car_y)
            sensor_end = (sensor_end_x, sensor_end_y)

            min_dist = self.sensor_range  # Default max sensor range

            # Query nearby boundary points using KD-Tree
            left_nearest_idx = self.left_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range)
            right_nearest_idx = self.right_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range)

            # Get nearby points
            left_nearby = list(zip(self.left_x[left_nearest_idx], self.left_y[left_nearest_idx]))
            right_nearby = list(zip(self.right_x[right_nearest_idx], self.right_y[right_nearest_idx]))

            # Combine left and right boundary points
            boundary_segments = left_nearby + right_nearby

            # Iterate over nearby boundary segments
            for i in range(len(boundary_segments) - 1):
                boundary_start = boundary_segments[i]
                boundary_end = boundary_segments[i + 1]

                intersection = self.line_intersection(sensor_start, sensor_end, boundary_start, boundary_end)
                if intersection:
                    # Compute distance to intersection point
                    dist = np.hypot(intersection[0] - car_x, intersection[1] - car_y)
                    if dist < min_dist:
                        min_dist = dist  # Update min distance

            readings.append(min_dist)  # Store shortest detected boundary distance

        return np.array(readings, dtype=np.float32)


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
        return (min_dist_left > self.car_radius) and (min_dist_right > self.car_radius)


    # draws environment in pygame
    def render(self):
        if self.render_mode != "human":
            return
        
        self.screen.fill((0, 255, 0)) #green background

        # Create a filled track area by combining left and right boundary points
        track_polygon = list(zip(self.left_x, self.left_y)) + list(zip(reversed(self.right_x), reversed(self.right_y)))
        pygame.draw.polygon(self.screen, (0, 0, 0), track_polygon)  # Fill track with white

        # Draw track boundaries
        pygame.draw.lines(self.screen, (255, 255, 255), closed=True, 
        points=list(zip(self.left_x, self.left_y)), width=2)
        pygame.draw.lines(self.screen, (255, 255, 255), closed=True, 
        points=list(zip(self.right_x, self.right_y)), width=2)

        # Draw checkpoints
        for i, (start, end) in enumerate(self.checkpoints):
            if not self.checkpoints_passed[i]:
                color = (0, 0, 255)
                pygame.draw.line(self.screen, color, start, end, 5)
            if i == len(self.checkpoints) - 1:
                color = (255, 0, 0)
                pygame.draw.line(self.screen, color, start, end, 5)

        # Draw sensors
        for i, distance in enumerate(self.sensor_data):
            sensor_angle = self.car_angle + self.sensor_angles[i]
            sensor_dx = np.cos(np.radians(sensor_angle))
            sensor_dy = np.sin(np.radians(sensor_angle))

            # sensor should keep the length
            sensor_x = self.car_position[0] + sensor_dx * self.sensor_range
            sensor_y = self.car_position[1] + sensor_dy * self.sensor_range

            # red for boundary, green for nothing
            color = (255, 0, 0) if distance != self.sensor_range else (0, 255, 0)
            pygame.draw.line(self.screen, color, self.car_position.astype(int), (int(sensor_x), int(sensor_y)), 2)

        # Draw car
        #pygame.draw.circle(self.screen, (0, 0, 255), self.car_position.astype(int), self.car_radius)
        self.draw_rotated_car()

        pygame.display.flip()
        self.clock.tick(30)

    def draw_rotated_car(self):
        """Rotates and draws the car image at its current position."""
        rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)  # Rotate counterclockwise
        car_rect = rotated_car.get_rect(center=self.car_position.astype(int))  # Position the image
        self.screen.blit(rotated_car, car_rect.topleft)  # Draw the car

    def reset(self):
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32)
        self.car_speed = 0
        self.car_angle = -90

        self.num_checkpoints = 50
        self.checkpoints = self.generate_checkpoints()

        self.current_checkpoint = 0  # Start at the first checkpoint
        self.checkpoints_passed = [False] * (self.num_checkpoints + 1)  # Extra for finish line
        return np.hstack(([self.car_speed], [self.car_angle / 180], self.get_sensor_readings()))

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
    