from typing import Dict
from dataclasses import dataclass
from argus.target import Target

@dataclass
class Observation: 
    discovered_targets: Dict[int, Target]
    
