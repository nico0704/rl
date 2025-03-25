import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class RaceCarEnv:

    def __init__(self, track_width=80, track_radius=300, render_mode="human"):
        # screen settings
        self.WIDTH, self.HEIGHT = 1920, 1000
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
        self.num_checkpoints = 80
        self.checkpoints = self.generate_checkpoints()

        # car
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32) #start position
        self.car_radius = 5
        self.car_angle = 120
        self.car_speed = 0
        self.MAX_SPEED = 5
        self.ACCELERATION = 0.5 #beschleunigung
        self.FRICTION = 0.05 #reibung
        self.TURN_SPEED = 3

        # Load the car image
        self.car_image = pygame.image.load("images/f1_top_view.png") 
        self.car_image = pygame.transform.scale(self.car_image, (30, 30)) 
        
        # sensors
        self.sensor_angles = np.linspace(-90, 90, 5, endpoint=True)
        self.sensor_range = 100
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

        self.surface_3d = pygame.Surface((960, self.HEIGHT))
        self.surface_2d = pygame.Surface((960, self.HEIGHT))
        self.triangle_steer = 0
        self.smoothed_triangle_angle = 0
        self.splitscreen = False
        self.surface_minimap = pygame.Surface((300, 300), pygame.SRCALPHA)  # Transparent surface

    def generate_wavy_loop(self, amplitude=40, frequency=5):
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
                # print(f'CHECKPOINT {self.current_checkpoint} CROSSED!')
                return self.current_checkpoint, False  # Reward for crossing a checkpoint

        # If all checkpoints are passed, activate finish line
        if self.current_checkpoint == self.num_checkpoints and not self.checkpoints_passed[-1]:
            start, end = self.checkpoints[-1]  # Finish line is at the last checkpoint
            if self.point_line_distance((car_x, car_y), start, end) < self.car_radius * 2:
                self.checkpoints_passed[-1] = True
                print('FINISH LINE CROSSED!')
                return 100.0, True  # Reward for completing the track
                

        return 0.0, False # No reward if no checkpoint was crossed
    
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

        # for visualization
        self.triangle_steer = steer
        smoothing_rate = 0.05  # lower = slower turning
        self.smoothed_triangle_angle += smoothing_rate * (self.triangle_steer - self.smoothed_triangle_angle)


        # Auto bewegen
        dx = np.cos(np.radians(self.car_angle)) * self.car_speed
        dy = np.sin(np.radians(self.car_angle)) * self.car_speed
        self.car_position += np.array([dx, dy])

        # Sensorwerte aktualisieren
        self.sensor_data = self.get_sensor_readings()

        # **Checkpoints belohnen (Belohnung zwischen 0 und 1 normalisieren)**
        checkpoint_reward, finished = self.check_checkpoint_crossed()
        checkpoint_reward /= self.num_checkpoints

        # **Geschwindigkeitsbelohnung normalisieren**
        if self.car_speed > 0:
            speed_reward = (self.car_speed / self.MAX_SPEED) * 0.75  # Vorwärts max 0.8 Belohnung
        elif self.car_speed < 0:
            speed_reward = (self.car_speed / self.MAX_SPEED) * 0.2  # Rückwärts max -0.3 Strafe
        else:
            speed_reward = -0.5  # Stillstand wird stärker bestraft

        # **Strafe für das Verlassen der Strecke**
        off_track = not self.is_on_track()
        track_penalty = -1.0 if off_track else 0.0  # Harte Strafe für das Verlassen der Strecke

        # **Finaler Reward: Kombiniere alle Belohnungen**
        reward = checkpoint_reward + speed_reward + track_penalty

        # **Status zurückgeben**
        return np.hstack(([self.car_speed], [self.car_angle / 180], self.sensor_data)), reward, off_track or finished

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

            # Query nearby indices from KD-Trees
            left_nearest_idx = sorted(self.left_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range))
            right_nearest_idx = sorted(self.right_boundary_tree.query_ball_point([car_x, car_y], self.sensor_range))

            for i in range(len(left_nearest_idx) - 1):
                idx1 = left_nearest_idx[i]
                idx2 = left_nearest_idx[i + 1]
                boundary_start = (self.left_x[idx1], self.left_y[idx1])
                boundary_end = (self.left_x[idx2], self.left_y[idx2])

                intersection = self.line_intersection(sensor_start, sensor_end, boundary_start, boundary_end)
                if intersection:
                    dist = np.hypot(intersection[0] - car_x, intersection[1] - car_y)
                    if dist < min_dist:
                        min_dist = dist

            for i in range(len(right_nearest_idx) - 1):
                idx1 = right_nearest_idx[i]
                idx2 = right_nearest_idx[i + 1]
                boundary_start = (self.right_x[idx1], self.right_y[idx1])
                boundary_end = (self.right_x[idx2], self.right_y[idx2])

                intersection = self.line_intersection(sensor_start, sensor_end, boundary_start, boundary_end)
                if intersection:
                    dist = np.hypot(intersection[0] - car_x, intersection[1] - car_y)
                    if dist < min_dist:
                        min_dist = dist

            readings.append(min_dist)

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


    def render(self):
        if self.render_mode != "human":
            return

        if self.splitscreen:
            # SPLITSCREEN MODE
            self.surface_3d.fill((135, 206, 235))  # Sky color
            self.surface_2d.fill((0, 255, 0))      # Grass background

            self.render_3d(self.surface_3d)
            self.render_2d(self.surface_2d)

            self.screen.blit(self.surface_3d, (0, 0))
            self.screen.blit(self.surface_2d, (960, 0))
        else:
            # SINGLE SCREEN MODE WITH MINIMAP
            self.screen.fill((135, 206, 235))  # Sky color
            self.render_3d(self.screen)

            # Draw minimap in top-right
            self.render_minimap(self.surface_minimap)
            self.screen.blit(self.surface_minimap, (self.WIDTH - 310, 10))  # 10px margin

        pygame.display.flip()
        self.clock.tick(30)


    def render_2d(self, surface):
        surface.fill((0, 255, 0))  # green background

        # Center the view on the middle of the track (not the car)
        center_x = np.mean(self.center_x)
        center_y = np.mean(self.center_y)
        view_center_x = 960 // 2
        view_center_y = self.HEIGHT // 2

        def offset_point(p):
            return (int(p[0] - center_x + view_center_x),
                    int(p[1] - center_y + view_center_y))

        # Draw track surface
        track_polygon = [offset_point(p) for p in zip(self.left_x, self.left_y)] + \
                        [offset_point(p) for p in zip(reversed(self.right_x), reversed(self.right_y))]
        pygame.draw.polygon(surface, (0, 0, 0), track_polygon)

        # Draw track boundaries
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [offset_point(p) for p in zip(self.left_x, self.left_y)], 2)
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [offset_point(p) for p in zip(self.right_x, self.right_y)], 2)

        # Draw checkpoints
        for i, (start, end) in enumerate(self.checkpoints):
            color = (0, 0, 255) if not self.checkpoints_passed[i] else (100, 100, 100)
            if i == len(self.checkpoints) - 1:
                color = (255, 0, 0)
            pygame.draw.line(surface, color, offset_point(start), offset_point(end), 5)

        # Draw sensors
        for i, distance in enumerate(self.sensor_data):
            sensor_angle = self.car_angle + self.sensor_angles[i]
            dx = np.cos(np.radians(sensor_angle))
            dy = np.sin(np.radians(sensor_angle))
            sensor_end = self.car_position + np.array([dx, dy]) * self.sensor_range
            color = (255, 0, 0) if distance != self.sensor_range else (0, 255, 0)
            pygame.draw.line(surface, color, offset_point(self.car_position), offset_point(sensor_end), 2)

        # Draw car and speed bar
        self.draw_rotated_car(surface, offset_point)
        self.draw_speed_bar(surface)



    def draw_rotated_car(self, surface, offset_point):
        """Draws the rotated car image at its current position, adjusted for camera offset."""
        # Rotate the car image based on current angle
        rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)

        # Offset the car position to center it in the 2D panel
        car_pos_offset = offset_point(self.car_position)

        # Get the new rect and draw it
        car_rect = rotated_car.get_rect(center=car_pos_offset)
        surface.blit(rotated_car, car_rect.topleft)
        def draw_rotated_car(self, surface):
            rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)
            car_rect = rotated_car.get_rect(center=self.car_position.astype(int))
            surface.blit(rotated_car, car_rect.topleft)



    def render_3d(self, surface):
        surface.fill((135, 206, 235))  # Sky color

        car_pos = self.car_position
        car_angle_rad = np.radians(self.car_angle)

        forward = np.array([np.cos(car_angle_rad), np.sin(car_angle_rad)])
        right = np.array([np.cos(car_angle_rad + np.pi / 2), np.sin(car_angle_rad + np.pi / 2)])

        # Adjust screen width based on mode
        screen_width = 960 if self.splitscreen else self.WIDTH
        screen_center_x = screen_width // 2

        scale_x = 300
        scale_y = 6000
        horizon_y = int(self.HEIGHT * 0.2)

        pygame.draw.rect(surface, (50, 200, 50), (0, horizon_y, screen_width, self.HEIGHT - horizon_y))

        # Closest point on centerline to car
        closest_idx = np.argmin(np.hypot(self.center_x - car_pos[0], self.center_y - car_pos[1]))

        prev_left = None
        prev_right = None

        for i in range(300):
            idx = (closest_idx + i * 4) % len(self.left_x)

            left_world = np.array([self.left_x[idx], self.left_y[idx]])
            right_world = np.array([self.right_x[idx], self.right_y[idx]])

            def to_camera(p):
                rel = p - car_pos
                cam_x = np.dot(rel, right)
                cam_y = np.dot(rel, forward)
                return cam_x, cam_y

            lx, ly = to_camera(left_world)
            rx, ry = to_camera(right_world)

            ly = max(ly, 1)
            ry = max(ry, 1)

            left_proj = (
                int(screen_center_x + (lx / ly) * scale_x),
                int(horizon_y + scale_y / ly)
            )
            right_proj = (
                int(screen_center_x + (rx / ry) * scale_x),
                int(horizon_y + scale_y / ry)
            )

            if prev_left and prev_right:
                pygame.draw.polygon(surface, (50, 50, 50), [
                    prev_left, prev_right, right_proj, left_proj
                ])

            prev_left = left_proj
            prev_right = right_proj

        self.draw_hud_car(surface)



    def draw_hud_car(self, surface):
        # Triangle size (controls how big the car triangle appears in 3D view)
        triangle_height = 150
        triangle_base = 300

        # Screen position: center bottom of the 3D panel (960px wide)
        center_x = self.WIDTH // 4 if self.splitscreen else self.WIDTH // 2 
        center_y = self.HEIGHT - 200  # a bit above the bottom edge

        # Define triangle points (pointing up)
        points = np.array([
            [0, -triangle_height],           # Tip (front)
            [-triangle_base / 2, 0],         # Bottom left
            [triangle_base / 2, 0]           # Bottom right
        ])

        # Convert smoothed steering input to rotation angle
        max_steer_angle = 60  # degrees
        steer_angle = np.clip(self.smoothed_triangle_angle, -1.0, 1.0) * max_steer_angle
        angle_rad = np.radians(steer_angle)

        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        rotated_points = points @ rotation_matrix.T

        # Translate to center position
        translated_points = rotated_points + np.array([center_x, center_y])

        # Draw the triangle
        pygame.draw.polygon(surface, (255, 0, 0), translated_points.astype(int))


    def draw_speed_bar(self, surface):
        bar_width = 30
        bar_height = 200
        bar_x = 960 - 50  # relative to the 2D panel
        bar_y = 700

        pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        zero_y = bar_y + bar_height // 2
        pygame.draw.line(surface, (255, 255, 255), (bar_x - 5, zero_y), (bar_x + bar_width + 5, zero_y), 2)

        normalized_speed = np.clip(self.car_speed / self.MAX_SPEED, -1, 1)
        fill_height = int(abs(normalized_speed) * (bar_height / 2))

        if self.car_speed > 0:
            fill_color = (200, 0, 0)
            fill_rect = (bar_x, zero_y - fill_height, bar_width, fill_height)
        elif self.car_speed < 0:
            fill_color = (0, 100, 255)
            fill_rect = (bar_x, zero_y, bar_width, fill_height)
        else:
            fill_color = (100, 100, 100)
            fill_rect = (bar_x, zero_y, bar_width, 2)

        pygame.draw.rect(surface, fill_color, fill_rect)


    def render_minimap(self, surface):
        surface.fill((0, 0, 0, 0))  # Clear with transparency

        # Scale down the track
        scale = 0.15
        center_x = np.mean(self.center_x)
        center_y = np.mean(self.center_y)
        mid_w, mid_h = surface.get_width() // 2, surface.get_height() // 2 - 50

        def scale_point(p):
            return (
                int((p[0] - center_x) * scale + mid_w),
                int((p[1] - center_y) * scale + mid_h)
            )

        # Draw track
        track_polygon = [scale_point(p) for p in zip(self.left_x, self.left_y)] + \
                        [scale_point(p) for p in zip(reversed(self.right_x), reversed(self.right_y))]
        pygame.draw.polygon(surface, (0, 0, 0), track_polygon)

        # Draw boundaries
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [scale_point(p) for p in zip(self.left_x, self.left_y)], 1)
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [scale_point(p) for p in zip(self.right_x, self.right_y)], 1)

        # Draw car as small red circle
        car_pos_scaled = scale_point(self.car_position)
        pygame.draw.circle(surface, (255, 0, 0), car_pos_scaled, 4)


    def reset(self):
        self.car_position = np.array([self.center_x[0], self.center_y[0]], dtype=np.float32)
        self.car_speed = 0
        
        # Get direction from centerline
        dx = self.center_x[0] - self.center_x[1]
        dy = self.center_y[0] - self.center_y[1]
        self.car_angle = np.degrees(np.arctan2(dy, dx))

        self.checkpoints = self.generate_checkpoints()

        self.current_checkpoint = 0  # Start at the first checkpoint
        self.checkpoints_passed = [False] * (self.num_checkpoints + 1)  # Extra for finish line
        return np.hstack(([self.car_speed], [self.car_angle / 180], self.get_sensor_readings()))

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
    