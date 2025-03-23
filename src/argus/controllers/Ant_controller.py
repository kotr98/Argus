from argus.drone import Drone
from argus.observation import Observation
from typing import Dict, List
import numpy as np
from argus.ACO.ant_colony import AntColony
from argus.graph.graph_distance_matrix import calculate_distance_matrix
from argus.graph.partition_graph import partition_pois_kmeans
from enum import Enum
from argus.poi import POI
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Literal
from argus.target import Target
from typing import Set

@dataclass
class DroneTask:
    action: str
    drone_id: int | None
    position: np.ndarray
    drone_type: Literal["attack", "surveillance"]

@dataclass
class TargetState:
    state: Literal["discovered", "attacking", "attacked"] #, "confirming", "confirmed"]

    def advance(self):
        if self.state == "discovered":
            self.state = "attacking"
        elif self.state == "attacking":
            self.state = "attacked"
        else:
            raise NotImplementedError()


class AntController():

    def __init__(self, drones: Dict[int, Drone], poi: List[POI], width: float, height: float):

        self.drones = drones
    
        self.poi = {}
        for p in poi:
            self.poi[p.id] = p

        self.width = width
        self.height = height

        self.drone_state_machines = {}
     
        self.target_tasks: Dict[int, DroneTask] = {}
        self.target_state: Dict[int, TargetState] = {}
        self.discovered_targets: Dict[int, Target] = {}
        self.taken_drones: Set[int]  = set()

        self.should_replan = False

        self._new_poi_idx = 100000

        self.replan()
       

    def replan(self):
        self.taken_drones = set()

        print("tasks: ", len(self.target_tasks))
        for target_id, task in list(self.target_tasks.items()):
            if task.drone_id is not None:
                if not self.drones[task.drone_id].is_idle():
                    
                    self.taken_drones.add(task.drone_id)

        self.replan_attack()


        free_surveilance_drones = {}
        for drone_id in self.drones.keys():
            if self.drones[drone_id].type != "surveillance":
                continue
            if drone_id in self.taken_drones:
                continue
            
            free_surveilance_drones[drone_id] = self.drones[drone_id]
            
        # check all drones and add them to the set again!
        self.replan_search(free_surveilance_drones)

    def replan_attack(self):

        for target_id, target_task in list(self.target_tasks.items()):
            if target_task.drone_id is None:
                drone_id = self.get_closest_drone(self.discovered_targets[target_id].position, target_task.drone_type, list(self.taken_drones))
                if drone_id is None:
                    print("Couldnt find fitting drone")
                    continue
                target_task.drone_id = drone_id
                self.drones[drone_id].fly_to_and_stop([POI(-2, target_task.position)])
                print(f"Starting attack and flying drone {drone_id} to {target_task.position}")
                self.taken_drones.add(drone_id)
            else:
                self.taken_drones.add(target_task.drone_id)
            

    def get_closest_drone(self, position: np.ndarray, type: str, excluding: List[int]) -> int:
        closest_drone_id = None 
        closest_distance = np.inf
        for drone_id, drone in self.drones.items():
            if drone_id in excluding:
                continue
            if drone.type != type:
                continue
            distance = np.linalg.norm(drone.pose[:2] - position)
            if distance < closest_distance:
                closest_distance = distance
                closest_drone_id = drone_id
        
        return closest_drone_id


    def replan_confirmation(self) -> List[int]:
        for target_id, task in self.target_tasks.items():
            task.drone_type == "surveilance"
        

    def replan_search(self, drones: Dict[int, Drone]):
        poi_list = list(self.poi.values())
        self.distances, self.node_coords = calculate_distance_matrix(poi_list)
        # Partition the graph into k clusters

        cluster_count = min(len(drones), len(self.poi))

        if cluster_count == 0:
            return

        poi_partitions = partition_pois_kmeans(poi_list, k=cluster_count)
        partition_distance_matrices = []
        # Calculate the distance matrix and node coordinates for each partition
        
        for poi_partition in poi_partitions:
            distance_matrix, _ = calculate_distance_matrix(poi_partition)
            partition_distance_matrices.append(distance_matrix)

        drone_list = list(drones.values())
        assignments = self.drone_cluster_assignment(poi_partitions, drone_list)

        assert len(assignments) == len(drones)

        for i in range(len(assignments)):
            waypoints = self.get_waypoints(drone_list[i], poi_partitions[assignments[i]], partition_distance_matrices[assignments[i]])
            drone_list[i].fly_to_and_stop(waypoints)


    # Returns the cluster index for each drone
    def drone_cluster_assignment(self, clusters: List[List[POI]], drones: List[Drone]) -> List[int]:

        
        drone_cluster_distance_matrix = np.zeros((len(drones), len(clusters)), dtype=np.float32)
        drone_cluster_distance_matrix[:,:] = np.inf

        for cluster_idx in range(len(clusters)):
            for drone_idx in range(len(drones)):
                drone_cluster_distance_matrix[drone_idx, cluster_idx] = self.get_drone_to_cluster_distance(clusters[cluster_idx], drones[drone_idx])

        row_indices, col_indices = linear_sum_assignment(drone_cluster_distance_matrix)

        assignments = [0]*len(drones)
        for i in range(len(row_indices)):
            assignments[row_indices[i]] = col_indices[i]

        return assignments

    def get_drone_to_cluster_distance(self, cluster: List[POI], drone: Drone):
        min_distance = np.inf
        for poi in cluster:
            distance = np.sqrt(np.sum((poi.position - drone.pose[:2])**2))
            if distance < min_distance:
                min_distance = distance
        
        return min_distance


        
    def process_observations(self, observations: Dict[int, Observation]):
        for drone_id, obs in observations.items():
            for target_id, target in obs.discovered_targets.items():
                if target_id in self.discovered_targets:
                    continue
                self.discovered_targets[target_id] = target
                self.target_state[target_id] = TargetState("discovered")

                print("Discovered target")

                

    def handle_target_tasks(self):
        for target_id, target_state in self.target_state.items():
            if target_state.state == "discovered":
                print("Launching attack")
                self.target_tasks[target_id] = DroneTask("attack", None, self.discovered_targets[target_id].position, "attack")
                target_state.advance()
                self.should_replan = True
            else:
                pass

        


    def step(self, dt: float, observations: Dict[int, Observation]):
        self.should_replan = False
        self.process_observations(observations)
        self.handle_target_tasks()

        for drone_id in self.drones.keys():
            self.prune_pois(drone_id)

        for target_id, task in list(self.target_tasks.items()):
            if task.drone_id is not None:
                if self.drones[task.drone_id].is_idle():
                    self.target_state[target_id].advance()
                    del self.target_tasks[target_id]
                    self.should_replan = True
                    print("Deleted task and scheduled replanning")
               


        for drone_id, drone in self.drones.items():
            if drone.type == "surveillance" and len(drone.waypoints) == 0:
                self.should_replan = True



        if self.should_replan:
            self.replan()


        for drone in self.drones.values():
            if len(drone.waypoints) == 0:
                drone.fly_to_and_stop([POI(-1,np.array([0.0,0.0]))])



    def prune_pois(self, drone_id: int):
        for finished_waypoint in self.drones[drone_id].finished_waypoints:
            if finished_waypoint.id in self.poi:
                del self.poi[finished_waypoint.id]



    def get_waypoints(self, drone: Drone, pois: List[POI], distances: np.array) -> List[POI]:
        if len(pois) <= 1:
            return pois
        
        ant_colony = AntColony(distances, 1, 1, 100, 0.95, alpha=1, beta=1) # ((distances, n_ants, n_best, n_iterations, decay, alpha: exponenet on pheromone, beta: exponent on distance))
        shortest_path = ant_colony.run()
    
        # Unpack the output from the ant colony.
        edges, total_cost = shortest_path  # total_cost is np.float64(9.0) in your example

        # Reconstruct the node order.
        node_order = [edges[0][0]]  # start with the starting node of the first edge
        for start, end in edges:
            node_order.append(end)
        

        waypoints = []

        for node in node_order:
            waypoints.append(pois[node])
        # Convert node indices to actual coordinates.
        #waypoints = [POI(node_coords[node]) for node in node_order]

        return waypoints