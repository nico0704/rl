#### The renderer which visualizes the environment ####
# 
# Usage:
#     via environment.py
# Description:
#     - Shows a realtime visualization showing the car driving the track which contains:
#       - A 3D visualization of the track, the checkpoints and the finish line
#       - A triangle as car showing the direction in which the car is driving
#       - A minimap in the top right corner showing a 2D top view of the track with a red dot as car indicator
#       - A speed bar visualizing the car speed
#       - A info box in the top left corner showing the current state [car speed, car angle, sensor data]

import pygame
import numpy as np

class Renderer:

    def __init__(self, width, height, car, track, checkpoints_passed):

        self.WIDTH = width
        self.HEIGHT = height
        self.car = car
        self.track = track
        self.checkpoints_passed = checkpoints_passed

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.triangle_steer = 0
        self.smoothed_triangle_angle = 0
        self.surface_minimap = pygame.Surface((np.clip(300 * (self.WIDTH / 1920), 0, 300), np.clip(300 * (self.HEIGHT / 1080), 0, 300)), pygame.SRCALPHA)  # transparent surface
        self.font = pygame.font.SysFont("Arial", int(np.clip(20 * (self.WIDTH / 1920), 0, 20)))


    def close(self):
        pygame.quit

    
    def render(self):

        self.screen.fill((135, 206, 235))  # sky color
        self.render_3d(self.screen)
        self.draw_speed_bar(self.screen)
        
        # draw minimap in top-right
        self.render_minimap(self.surface_minimap)
        self.screen.blit(self.surface_minimap, (self.WIDTH - np.clip(310 * (self.WIDTH / 1920), 0, 310), 10))  # 10px margin

        self.draw_info_box(self.screen)

        pygame.display.flip()
        self.clock.tick(30)

            
    def render_3d(self, surface):

        car_pos = self.car.car_position
        car_angle_rad = np.radians(self.car.car_angle)

        forward = np.array([np.cos(car_angle_rad), np.sin(car_angle_rad)])
        right = np.array([np.cos(car_angle_rad + np.pi / 2), np.sin(car_angle_rad + np.pi / 2)])

        # adjust screen width based on mode
        screen_center_x = self.WIDTH // 2

        scale_x = 300
        scale_y = 6000
        horizon_y = int(self.HEIGHT * 0.2)

        pygame.draw.rect(surface, (50, 200, 50), (0, horizon_y, self.WIDTH, self.HEIGHT - horizon_y))

        # closest point on centerline to car
        closest_idx = np.argmin(np.hypot(self.track.center_x - car_pos[0], self.track.center_y - car_pos[1]))

        prev_left = None
        prev_right = None
        start, end = self.track.checkpoints[-1]

        for i in range(300):
            idx = (closest_idx + i * 4) % len(self.track.left_x)

            left_world = np.array([self.track.left_x[idx], self.track.left_y[idx]])
            right_world = np.array([self.track.right_x[idx], self.track.right_y[idx]])

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
            
            lx, ly = to_camera(np.array(start))
            rx, ry = to_camera(np.array(end))

            if ly > 1 and ry > 1:
                left_proj = (
                    int(screen_center_x + (lx / ly) * scale_x),
                    int(horizon_y + scale_y / ly)
                )
                right_proj = (
                    int(screen_center_x + (rx / ry) * scale_x),
                    int(horizon_y + scale_y / ry)
                )

                pygame.draw.line(surface, (255, 255, 255), left_proj, right_proj, 2)
        
        for i, (start, end) in enumerate(self.track.checkpoints[:-1]):  # skip last (finish line)
            if self.checkpoints_passed[i] == False:
                def to_camera(p):
                    rel = p - car_pos
                    cam_x = np.dot(rel, right)
                    cam_y = np.dot(rel, forward)
                    return cam_x, cam_y

                sx, sy = to_camera(np.array(start))
                ex, ey = to_camera(np.array(end))

                if sy > 1 and ey > 1:
                    start_proj = (
                        int(screen_center_x + (sx / sy) * scale_x),
                        int(horizon_y + scale_y / sy)
                    )
                    end_proj = (
                        int(screen_center_x + (ex / ey) * scale_x),
                        int(horizon_y + scale_y / ey)
                    )

                    pygame.draw.line(surface, (0, 162, 250), start_proj, end_proj, 2)

                
        self.draw_hud_car(surface)


    def draw_hud_car(self, surface):
        # triangle size (controls how big the car triangle appears in 3D view)
        triangle_height = np.clip(150 * (self.WIDTH / 1920), 0, 150)
        triangle_base = np.clip(300 * (self.HEIGHT / 1080), 0, 300)

        # screen position: center bottom of the 3D panel (960px wide)
        center_x = self.WIDTH // 2 
        center_y = self.HEIGHT - np.clip(200 * (self.HEIGHT / 1080), 0, 200)  # a bit above the bottom edge

        # define triangle points (pointing up)
        points = np.array([
            [0, -triangle_height],           # tip (front)
            [-triangle_base / 2, 0],         # bottom left
            [triangle_base / 2, 0]           # bottom right
        ])

        # convert smoothed steering input to rotation angle
        max_steer_angle = 60  # degrees
        steer_angle = np.clip(self.smoothed_triangle_angle, -1.0, 1.0) * max_steer_angle
        angle_rad = np.radians(steer_angle)

        # rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        rotated_points = points @ rotation_matrix.T

        # translate to center position
        translated_points = rotated_points + np.array([center_x, center_y])

        # draw the triangle
        pygame.draw.polygon(surface, (255, 0, 0), translated_points.astype(int))


    def draw_speed_bar(self, surface):
        bar_width = np.clip(30 * (self.WIDTH / 1920), 0, 30)
        bar_height = np.clip(600 * (self.HEIGHT / 1080), 0, 600)
        bar_x = surface.get_width() - np.clip(100 * (self.WIDTH / 1920), 0, 100)
        bar_y = np.clip(300 * (self.HEIGHT / 1080), 0, 300)

        pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        zero_y = bar_y + bar_height // 2
        pygame.draw.line(surface, (255, 255, 255), (bar_x - 5, zero_y), (bar_x + bar_width + 5, zero_y), 2)

        normalized_speed = np.clip(self.car.car_speed / self.car.MAX_SPEED, -1, 1)
        fill_height = int(abs(normalized_speed) * (bar_height / 2))

        if self.car.car_speed > 0:
            fill_color = (200, 0, 0)
            fill_rect = (bar_x, zero_y - fill_height, bar_width, fill_height)
        elif self.car.car_speed < 0:
            fill_color = (0, 100, 255)
            fill_rect = (bar_x, zero_y, bar_width, fill_height)
        else:
            fill_color = (100, 100, 100)
            fill_rect = (bar_x, zero_y, bar_width, 2)

        pygame.draw.rect(surface, fill_color, fill_rect)


    def render_minimap(self, surface):
        surface.fill((0, 0, 0, 0))  # clear with transparency

        # scale down the track
        scale = np.clip(0.15 * (self.HEIGHT / 1080), 0, 0.15)
        center_x = np.mean(self.track.center_x)
        center_y = np.mean(self.track.center_y)
        mid_w, mid_h = surface.get_width() // 2 - np.clip(50 * (self.WIDTH / 1920), 0, 50), surface.get_height() // 2 - np.clip(50 * (self.HEIGHT / 1080), 0, 50)

        def scale_point(p):
            return (
                int((p[0] - center_x) * scale + mid_w),
                int((p[1] - center_y) * scale + mid_h)
            )

        # draw track
        track_polygon = [scale_point(p) for p in zip(self.track.left_x, self.track.left_y)] + \
                        [scale_point(p) for p in zip(reversed(self.track.right_x), reversed(self.track.right_y))]
        pygame.draw.polygon(surface, (0, 0, 0), track_polygon)

        # draw boundaries
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [scale_point(p) for p in zip(self.track.left_x, self.track.left_y)], 1)
        pygame.draw.lines(surface, (255, 255, 255), True,
                        [scale_point(p) for p in zip(self.track.right_x, self.track.right_y)], 1)

        # draw car as small red circle
        car_pos_scaled = scale_point(self.car.car_position)
        pygame.draw.circle(surface, (255, 0, 0), car_pos_scaled, 4)


    def draw_info_box(self, surface):
        box_x, box_y = np.clip(20 * (self.WIDTH / 1920), 0, 20), np.clip(20 * (self.HEIGHT / 1080), 0, 20)
        box_width, box_height = np.clip(280 * (self.WIDTH / 1920), 0, 280), np.clip(190 * (self.HEIGHT / 1080), 0, 190)
        padding = np.clip(10 * (self.HEIGHT / 1080), 0, 10)

        # background box
        pygame.draw.rect(surface, (30, 30, 30), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(surface, (255, 255, 255), (box_x, box_y, box_width, box_height), 2)


        speed_text = f"Speed: {self.car.car_speed:.2f}"
        angle_text = f"Angle: {self.car.car_angle:.2f}"
        sensor_texts = [f"S{i+1}: {s:.1f}" for i, s in enumerate(self.car.sensor_data)]

        lines = [speed_text, angle_text] + sensor_texts

        # render lines
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            surface.blit(text_surface, (box_x + padding, box_y + padding + i * np.clip(25 * (self.HEIGHT / 1080), 0, 25)))
            

    def step(self, action):
        _, steer = action
        self.triangle_steer = steer
        smoothing_rate = 0.05  # lower = slower turning
        self.smoothed_triangle_angle += smoothing_rate * (self.triangle_steer - self.smoothed_triangle_angle)


    
