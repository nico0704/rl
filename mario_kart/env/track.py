import numpy as np
from scipy.spatial import KDTree

class Track:

    def __init__(self, width, height, track_width, track_radius, num_checkpoints, num_points):
        
        # generate track
        self.WIDTH = width
        self.HEIGHT = height
        self.track_width = track_width
        self.track_radius = track_radius
        self.num_points = num_points
        self.num_checkpoints = num_checkpoints

        self.center_x, self.center_y = self.generate_wavy_loop(
            amplitude=np.random.randint(30, 70), 
            frequency=np.random.randint(2,7)
        )
        self.left_x, self.left_y, self.right_x, self.right_y = self.compute_boundaries()

        # generate kd-trees for boundary points
        self.left_boundary_tree = KDTree(np.column_stack((self.left_x, self.left_y)))
        self.right_boundary_tree = KDTree(np.column_stack((self.right_x, self.right_y)))

        self.checkpoints = self.generate_checkpoints()


    def generate_wavy_loop(self, amplitude=40, frequency=5):

        angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        np.append(angles, 0.0)
        center_x = self.WIDTH // 2 + (self.track_radius + amplitude * np.sin(frequency * angles)) * np.cos(angles)
        center_y = self.HEIGHT // 2 + (self.track_radius + amplitude * np.sin(frequency * angles)) * np.sin(angles)
        return center_x, center_y


    def compute_boundaries(self):

        left_x, left_y = [], []
        right_x, right_y = [], []

        num_points = len(self.center_x)

        for i in range(num_points):
            # compute tangent (direction)
            dx = self.center_x[(i + 1) % num_points] - self.center_x[i]
            dy = self.center_y[(i + 1) % num_points] - self.center_y[i]

            # normalize tangent
            length = np.hypot(dx, dy)
            dx /= length
            dy /= length

            # compute perpendicular vector
            perp_x = -dy
            perp_y = dx

            # offset points to get left and right boundaries
            half_width = self.track_width / 2
            left_x.append(self.center_x[i] + half_width * perp_x)
            left_y.append(self.center_y[i] + half_width * perp_y)
            right_x.append(self.center_x[i] - half_width * perp_x)
            right_y.append(self.center_y[i] - half_width * perp_y)

        return np.array(left_x), np.array(left_y), np.array(right_x), np.array(right_y)


    def generate_checkpoints(self):

        checkpoints = []
        step = len(self.center_x) // self.num_checkpoints

        for i in range(self.num_checkpoints):
            idx = i * step
            dx = self.center_x[(idx + 1) % len(self.center_x)] - self.center_x[idx]
            dy = self.center_y[(idx + 1) % len(self.center_y)] - self.center_y[idx]

            # normalize
            length = np.hypot(dx, dy)
            dx /= length
            dy /= length

            # compute perpendicular
            perp_x = -dy
            perp_y = dx

            # create checkpoint line
            half_width = self.track_width / 2
            checkpoint_start = (self.center_x[idx] + half_width * perp_x, self.center_y[idx] + half_width * perp_y)
            checkpoint_end = (self.center_x[idx] - half_width * perp_x, self.center_y[idx] - half_width * perp_y)

            checkpoints.append((checkpoint_start, checkpoint_end))

        # move the first checkpoint to the end of the list (finish line)
        return list(reversed(checkpoints))
