import networkx as nx
from sklearn.cluster import KMeans
from networkx.classes.reportviews import NodeView
import numpy as np
from typing import List
from argus.poi import POI

def partition_graph_mst(G, k, weight='weight'):
    """
    Partition graph G into k connected components by computing the MST and
    removing the k-1 most expensive edges.

    Parameters:
      G (networkx.Graph): The original graph.
      k (int): Number of desired connected components.
      weight (str): The edge attribute that holds the numerical value used for
                    computing the MST (default 'weight').

    Returns:
      List[set]: A list of sets, each containing the nodes of one connected component.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    # Compute the minimum spanning tree (MST) of the graph
    mst = nx.minimum_spanning_tree(G, weight=weight)

    # If k==1, return the entire MST as a single component
    if k == 1:
        return [set(mst.nodes())]

    # Get all edges from the MST with their data, and sort them by weight in descending order
    edges_sorted = sorted(mst.edges(data=True), key=lambda x: x[2].get(weight, 1), reverse=True)

    # Remove the k-1 edges with the highest weight
    for i in range(min(k - 1, len(edges_sorted))):
        u, v, data = edges_sorted[i]
        mst.remove_edge(u, v)
    return mst

    # Return the connected components from the modified MST
    return list(nx.connected_components(mst))



def partition_pois_kmeans(pois: List[POI], k: int) -> List[List[POI]]:
    """
    Partitions the given points of interest (POIs) into k clusters using KMeans.
    
    Parameters:
        pois (List[POI]): A list of POI objects.
        k (int): The number of clusters.
        
    Returns:
        List[List[POI]]: A list of clusters, each of which is a list of POI objects.
    """
    # Extract coordinates from the POIs for clustering
    coordinates = np.array([poi.position for poi in pois])
    
    # Fit KMeans on the coordinates
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(coordinates)
    
    labels = kmeans.labels_
    clusters = [[] for _ in range(k)]
    
    # Group each POI by its cluster label
    for poi, label in zip(pois, labels):
        clusters[label].append(poi)
    
    return clusters
    