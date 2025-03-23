from argus.environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from argus.controllers.Ant_controller import AntController

class Visualization:
    
    def __init__(self, env: Environment):
        self.env = env
        # Create a figure and axis once
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.ion()  # Enable interactive mode
        self.colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']

    def draw_env(self):
        # Clear previous frame
        self.ax.clear()

        # Plot the satellite image as the background
        self.ax.imshow(self.env.sat_array)

        
        self.ax.set_xlim(0, self.env.grid_width)
        self.ax.set_ylim(0, self.env.grid_height)
        self.ax.set_aspect('equal')
        
        # Plot drones as green circles with FOV lines
        for drone_id, drone in self.env.drones.items():
            x, y, theta = drone.pose
            waypoints = drone.waypoints

            # Compute FOV boundaries.
            fov_rad = 1.39
            left_angle = theta - fov_rad / 2
            right_angle = theta + fov_rad / 2
            
            # Define a length for the FOV lines.
            ray_length = 50  
            # Calculate endpoints for both FOV boundary lines.
            x_left = x + ray_length * np.cos(left_angle)
            y_left = y + ray_length * np.sin(left_angle)
            x_right = x + ray_length * np.cos(right_angle)
            y_right = y + ray_length * np.sin(right_angle)
            
            if len(drone.waypoints)==0 or drone.waypoints[0].id == -1:
                drone_color = "gray"
            else:
                drone_color = self.colors[drone_id]
            # Draw the drone as a filled circles.
            
            if drone.type == "attack":
                drone_circle = plt.Circle((x,y), radius=30.0, fill=False, edgecolor='red', linewidth=2)

            else:
                drone_circle = plt.Circle((x,y), radius=10.0, color=drone_color, fill=True)

            # Plot the FOV boundary lines.
            self.ax.plot([x, x_left], [y, y_left], color=drone_color, linestyle='-', linewidth=2)
            self.ax.plot([x, x_right], [y, y_right], color=drone_color, linestyle='-', linewidth=2)
            

            self.ax.add_patch(drone_circle)
            

            # Plot waypoints as crosses.
            for waypoint in waypoints:
                x, y = waypoint.position
                self.ax.plot(x, y, marker='x', markersize=10, color=self.colors[drone_id])


            controller: AntController = self.env.controller
            # only draw the target if it is discovered
            for target_id, target in controller.discovered_targets.items():
                x, y = target.position
                if controller.target_state[target_id].state == "discovered":
                    self.ax.plot(x, y, 'g*', markersize=20)
                elif controller.target_state[target_id].state == "attacking":
                    self.ax.plot(x,y, 'r*', markersize=20)
                elif controller.target_state[target_id].state == "attacked":
                    self.ax.plot(x, y, "k*", markersize=20)
                else:
                    raise NotImplementedError()
        
            '''
            # Plot targets as red dots. (Permanent)
            for _, target in self.env.targets.items():
                x, y = target.position
                self.ax.plot(x, y, 'y*', markersize=10)
            '''

        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

