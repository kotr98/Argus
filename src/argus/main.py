import yaml
import numpy as np
import time
import matplotlib.pyplot as plt
import osmnx as ox
import random

from argus.environment import Environment
from argus.drone import Drone
from argus.target import Target
from argus.controllers.Ant_controller import AntController
from argus.visualization import Visualization
from argus.graph.load_graph import load_graph, BoundingBox
from argus.graph.utils import graph_to_coords
from argus.graph.download_sat import download_satellite_image
from argus.poi import POI


def load_config(config_path="./configs/config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_env(config):
    # Use bounding box coordinates from config
    north = config["north"]
    west = config["west"]
    south = config["south"]
    east = config["east"]

    # Build drones dictionary from config
    drones = {}
    for key, val in config["drones"].items():
        # Each entry: [type, [starting coordinates], speed]
        drone_type, start_coords, speed = val
        drones[int(key)] = Drone(drone_type, np.array(start_coords), speed)

    print("Download, save and load graph")
    # Get the graph for the area (OSMnx expects: west, south, east, north)
    G = ox.graph_from_bbox([west, south, east, north], network_type='drive')
    ox.save_graphml(G, "./tmp/graph.graphml_tmp")
    map_obj = load_graph("./tmp/graph.graphml_tmp", BoundingBox(north, west, south, east))

    # Generate POIs from the graph coordinates
    coords = graph_to_coords(map_obj.graph)
    pois = [POI(i, coord) for i, coord in enumerate(coords)]

    # Create targets either by random selection or using manual coordinates from config
    targets = {}
    if config.get("random_targets", False):
        num_targets = config.get("number_of_targets", 1)
        indices = random.sample(range(coords.shape[0]), num_targets)
        for i, idx in enumerate(indices):
            targets[i] = Target(coords[idx])
    else:
        for key, coord in config.get("targets", {}).items():
            targets[key] = Target(np.array(coord))

    # Initialize the controller with the drones and POIs
    controller = AntController(drones, pois, map_obj.width, map_obj.height)

    print("Downloading satellite image")
    sat_array = download_satellite_image(west, south, east, north, map_obj.width, map_obj.height)

    return Environment(
        width=map_obj.width,
        height=map_obj.height,
        cell_size=1.0,
        drones=drones,
        targets=targets,
        controller=controller,
        sat_array=sat_array
    )


def main(config=None):
    # Use provided config or load from the default file
    if config is None:
        config = load_config()
    env = setup_env(config)
    visualizer = Visualization(env)

    dt = 0.01  # seconds
    index = 0

    while True:
        index += 1
        env.step(dt)
        time.sleep(dt)

        # Render every 5th frame for performance
        if index % 5 == 0:
            visualizer.draw_env()
            plt.pause(dt)


if __name__ == "__main__":
    main()