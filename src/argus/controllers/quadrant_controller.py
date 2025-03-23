from argus.drone import Drone
from argus.observation import Observation
from typing import Dict, List
import numpy as np

class QuadrantController():

    def __init__(self, drones: Dict[int, Drone]):
        self.drones = drones

        # Define quadrant boundaries as tuples: (x_min, x_max, y_min, y_max)
        quadrants = [
            (0, 50, 50, 100),   # Drone 0: Upper Left
            (50, 100, 50, 100),  # Drone 1: Upper Right
            (0, 50, 0, 50),      # Drone 2: Lower Left
            (50, 100, 0, 50)     # Drone 3: Lower Right
        ]
        
        # Generate waypoints for each drone
        for i, drone in self.drones.items():
            x_min, x_max, y_min, y_max = quadrants[i]
            
            # Create 6 evenly spaced points in each dimension (6x6 grid = 36 waypoints)
            xs = np.linspace(x_min, x_max, 6)
            ys = np.linspace(y_min, y_max, 6)
            
            waypoints = []
            # Create a boustrophedon path: alternate the order of x values for consecutive rows
            for row_index, y in enumerate(ys):
                if row_index % 2 == 0:
                    x_order = xs
                else:
                    x_order = xs[::-1]
                for x in x_order:
                    waypoints.append(np.array([x, y]))
            
            # Add the computed waypoints to the drone
            drone.fly_to(waypoints)
    def step(self, dt: float, observations: Dict[int, Observation]):
        pass
