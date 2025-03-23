import osmnx as ox

from geopy.distance import geodesic
import numpy as np
from networkx.classes.reportviews import NodeView
from networkx import MultiDiGraph
import numpy as np
from typing import Tuple, List
from argus.poi import POI

#G = ox.load_graphml("./graph.graphml")


def calculate_distance_matrix(pois: List[POI]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the pairwise Euclidean distance matrix for a set of POIs and returns 
    both the distance matrix and the coordinates.

    Parameters:
        pois (List[POI]): A list of POI objects.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - distances: A matrix of shape (n_points, n_points) where distances[i, j] is the 
                         Euclidean distance between POI i and POI j. Self-distances (zeros) are 
                         replaced with np.inf.
            - coordinates: A np.float32 array of shape (n_points, 2) containing the POI coordinates.
    """
    n_points = len(pois)
    distances = np.zeros((n_points, n_points), dtype=np.float32)
    coordinates = np.zeros((n_points, 2), dtype=np.float32)
    
    # Build the coordinates array from each POI's position
    for i, poi in enumerate(pois):
        coordinates[i, :] = poi.position.astype(np.float32)
    
    # Compute pairwise Euclidean distances
    for i in range(n_points):
        for j in range(n_points):
            dx = coordinates[i, 0] - coordinates[j, 0]
            dy = coordinates[i, 1] - coordinates[j, 1]
            distances[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Replace self-distances (zeros) with infinity
    distances[distances == 0] = np.inf
    
    return distances, coordinates



#ox.plot_graph(G)