
import osmnx as ox
import networkx as nx
from networkx.classes.reportviews import NodeView
import numpy as np
from geopy.distance import distance
from dataclasses import dataclass

@dataclass
class BoundingBox:
    north: float 
    west: float 
    south: float 
    east: float

@dataclass
class Map:
    graph: nx.MultiDiGraph
    width: float 
    height: float

def load_graph(path: str, bb: BoundingBox) -> Map:
    G = ox.load_graphml(path)

    nodes = list(G.nodes(data=True))

    coordinates = np.zeros((len(nodes), 2), dtype=np.float64)

    for i, (_, data) in enumerate(nodes):
            coordinates[i, 0] = data["x"]
            coordinates[i, 1] = data["y"]

    x_min, x_max = bb.west, bb.east 
    y_min, y_max = bb.south, bb.north

    x_dist = x_max - x_min
    y_dist = y_max-y_min

    x_dist_meters = distance((x_min, y_min), (x_max, y_min)).meters
    y_dist_meters = distance((x_min, y_min), (x_min, y_max)).meters

    multiplier_x = x_dist_meters/x_dist
    multiplier_y = y_dist_meters/y_dist

    coordinates[:, 0] -= x_min 
    coordinates[:, 1] -= y_min 
    
    coordinates[:, 0] *= multiplier_x
    coordinates[:, 1] *= multiplier_y

    for i, (_, data) in enumerate(nodes):
            data["x"] = coordinates[i, 0]
            data["y"] = coordinates[i, 1]

    map = Map(
          graph=G,
          width=(x_max-x_min)*multiplier_x,
          height=(y_max-y_min)*multiplier_y,
    )


    return map
