import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

class System(ABC):
    def __init__(self, dim_state, dim_action):
        self.dim_state = dim_state  # Anzahl Zustandsvariablen/ Dimension des Zustandsraums
        self.dim_action = dim_action # Anzahl Steuerungsvariablen/ Dimension des Aktionsraums

    @abstractmethod
    def get_next_state(self, state, action): # gibt nächsten Zustand passierend auf Vorherigen und Aktion
        pass

class ThreeWheelRobot(System):
    def __init__(self, time_step_size, action_bounds, noise_power):
        super().__init__(dim_state=3, dim_action=2)     # 3D Zustandssystem (Position x,y und Orientierung) und 2D Aktionssystem (Translation- und Rotation)
        self.time_step_size = time_step_size # Größe des Zeitschrittes (dt)
        self.action_bounds = np.array(action_bounds)  # Grenzen/Intervalle für Translation- und Rotationsgeschwindigkeit - shape (2,)
        self.noise_power = noise_power # Rauschstärke für Zustandsänderung

    def get_next_state(self, current_state, action): # Zustandsberechnung
        # Ensure action is within bounds
        # print("before", action)
        # action = np.clip(action, self.action_bounds[0], self.action_bounds[1]) # Action bleibt im Interval
        # print("before", action)
        
        # Unpack state and action
        x, y, theta = current_state
        translational_velocity, angular_velocity = action # Translations- und Rotationsgeschwindigkeit aus Aktion

        # Time step
        dt = self.time_step_size

        # Compute noise 
        noise = np.random.normal(0, np.sqrt(self.noise_power * dt ** 2), size=3) # Zufälliges Rauschen mit Normalverteilung (3 Werte für x,y und orientierung)
        # noise = np.abs(noise)

        # Compute new state using Euler integration
        x_next = x + translational_velocity * np.cos(theta) * dt + noise[0] 
        y_next = y + translational_velocity * np.sin(theta) * dt + noise[1]
        theta_next = theta + angular_velocity * dt + noise[2]

        # Return the next state
        return np.array([x_next, y_next, theta_next])

    def compute_action_to_goal(self, state, goal):
        x, y, theta = state
        goal_x, goal_y = goal

        # Calculate distance and angle to goal
        distance_to_goal = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)

        # Calculate error in orientation
        orientation_error = angle_to_goal - theta
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error)) # normalize angle
        
        # Proportional control gains
        # translational_gain = 5.0  # speed gain
        # angular_gain = 5.0 # rotation speed gain
        
        # Control actions with control gains
        # translational_velocity = translational_gain * distance_to_goal
        # angular_velocity = angular_gain * orientation_error

        # Control actions
        translational_velocity = distance_to_goal
        angular_velocity = orientation_error
        
        # Ensure actions are within bounds
        translational_velocity = np.clip(translational_velocity, self.action_bounds[0][0], self.action_bounds[0][1])
        angular_velocity = np.clip(angular_velocity, self.action_bounds[1][0], self.action_bounds[1][1])
        # print(translational_velocity, angular_velocity)
        return np.array([translational_velocity, angular_velocity])

# Instantiate a three-wheel robot

# Define parameters
time_step_size = 0.01  # 10 ms
action_bounds = [-0.22, 0.22], [-2.84, 2.84]  # Translational and angular velocity bounds
noise_power = 0.8  # Example value for noise power
goal = np.array([0, 0])  # Goal position (0, 0)

# Instantiate the ThreeWheelRobot
robot = ThreeWheelRobot(time_step_size, action_bounds, noise_power)

# Simulation Loop

# Initialize state (e.g., starting at the origin with zero orientation)
state = np.array([2.0, 2.0, np.pi/2]) # Roboter startet an Koordinaten [2,2] und pi/2 als Orientierung

# Initialize history lists
state_history = [state.copy()]  # Alle Zustände
action_history = []     # Alle Aktionen

# Simulation parameters
simulation_time = 40  # Total simulation time in seconds
# time_step_size = 0.1  # Time step size in seconds
num_steps = int(simulation_time / time_step_size) # Führt Simulation num_steps oft durch
print("Simulation steps: ", num_steps)

# Run the simulation loop
for i in range(num_steps):
    # compute action
    action = robot.compute_action_to_goal(state, goal)

    # Update state using the ThreeWheelRobot's get_next_state method
    state = robot.get_next_state(state, action)

    # Save the current state and action to the history
    state_history.append(state.copy())
    action_history.append(action.copy())

    # Check if goal is reached
    if np.linalg.norm(state[:2] - goal) < 0.01:  # Check if within 0.1 units of the goal
        print(f"Goal reached after {i} steps at state: {state}")
        break

# Convert histories to numpy arrays for easier analysis
state_history = np.array(state_history)
action_history = np.array(action_history)

# Print the final state for reference
print("Final state:", state)

# Plot robot's trajectory
# Visualisierung der Daten

# Extract x, y coordinates and orientations from the state history
x_coords = state_history[:, 0]
y_coords = state_history[:, 1]
orientations = state_history[:, 2]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, label="Trajectory", alpha=0.7)

# Plot directed markers at regular intervals to show orientation
interval = max(1, len(x_coords) // 25)  # Plot every 50th point (or fewer if necessary)
for i in range(0, len(x_coords), interval): # interval als Supportindex
    x = x_coords[i]
    y = y_coords[i]
    theta = orientations[i]

    # Compute the arrow direction
    arrow_length = 0.00000001  # Length of the arrow representing orientation
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)

    plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, fc='red', ec='red')

# Mark the goal position
plt.plot(goal[0], goal[1], 'go', label=f"Goal {goal[0]},{goal[1]}")

# Labels and legend
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Robot Trajectory with Orientation Markers")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
