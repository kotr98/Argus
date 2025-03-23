import numpy as np

class Target:
    def __init__(self, position: np.ndarray):
        self.position = position

    def step(self, dt: float):
        pass

    def render(self, grid: np.ndarray):
        pass
        