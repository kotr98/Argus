import numpy as np
from typing import List
import math
from argus.poi import POI
from typing import Literal

class Drone:

    def __init__(self, type: Literal["attack", "surveillance"], pose: np.ndarray, speed: float):
        # [x,y,theta]
        self.type = type
        self.pose = pose
        self.waypoints: List[POI] = [] #[x,y]
        self.finished_waypoints: List[POI] = []
        self.speed = speed
        
        
    def _get_current_direction_vector(self):
        direction =  np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
        return direction / np.linalg.norm(direction)

    def _currently_at_waypoint(self) -> bool:
        if len(self.waypoints) == 0:
            return False

        delta = self.pose[:2] - self.waypoints[0].position
        return np.linalg.norm(delta) < 100
    

    def _update_position(self, dt: float):
        if len(self.waypoints) > 0:
            delta = self.waypoints[0].position - self.pose[:2]
            norm = np.linalg.norm(delta)
            if norm < 0.001:
                return
            delta_normed = delta / norm
            new_orientation = math.atan2(delta[1], delta[0])
            new_position = self.pose[:2] + delta_normed*dt*self.speed
            self.pose = np.array([new_position[0], new_position[1], new_orientation])

    def _update_waypoints(self):
        if self._currently_at_waypoint():
            self.finished_waypoints.append(self.waypoints.pop(0))
        

    def step(self, dt: float):
        self._update_position(dt)
        self._update_waypoints()

    def fly_to_and_stop(self, waypoints: List[POI]):
        self.waypoints = waypoints

    def is_idle(self) -> bool:
        return len(self.waypoints) == 0 or self.waypoints[0].id == -1
        
