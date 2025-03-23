import numpy as np
from dataclasses import dataclass

@dataclass
class POI:
    id: int
    position: np.ndarray
    target_id: int | None = None