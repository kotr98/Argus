
import osmnx as ox
import networkx as nx
from networkx.classes.reportviews import NodeView
import numpy as np
from geopy.distance import distance

def graph_to_coords(graph: nx.MultiDiGraph) -> np.ndarray:
    nodes = list(graph.nodes(data=True))

    coordinates = np.zeros((len(nodes), 2), dtype=np.float64)
    for i, (_, data) in enumerate(nodes):
            coordinates[i, 0] = data["x"]
            coordinates[i, 1] = data["y"]
    
    return coordinates