import numpy as np
from argus.target import Target
from typing import List, Dict
from argus.drone import Drone
from argus.observation import Observation
from argus.controllers.Ant_controller import AntController

class Environment:

    def __init__(self, width: int, height: int, cell_size: float, drones: Dict[int, Drone], targets: Dict[int, Target], controller: AntController, sat_array: np.ndarray):
        self.grid_width = int(width/cell_size)
        self.grid_height = int(height/cell_size)
        self.cell_size = cell_size
        self._grid = np.zeros((1, self.grid_height, self.grid_width))
        self.targets: Dict[int, Target] = targets
        self.drones: Dict[int, Drone] = drones
        self.controller = controller
        self.sat_array = sat_array

    
    def get_sensor_observation(self, drone_id: int) -> Observation:
        targets = self.get_discovered_targets(drone_id)
        return Observation(targets)

    def get_discovered_targets(self, drone_id: int):
        drone = self.drones[drone_id]
        targets = {}
        for target_id, target in self.targets.items():
            distance = np.sqrt(np.sum((drone.pose[:2]-target.position)**2))
            if distance < 100:
                targets[target_id]  = self.targets[target_id]
        
        return targets
    
    def step(self, dt: float):
        
        for _, target in self.targets.items():
            target.step(dt)
        
        
        for _, drone in self.drones.items():
            drone.step(dt)

        observations = {}
        for drone_id, drone in self.drones.items():
            observations[drone_id] = self.get_sensor_observation(drone_id)

        self.controller.step(dt, observations)


    