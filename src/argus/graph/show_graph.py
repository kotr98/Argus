import osmnx as ox
from argus.graph.partition_graph import partition_graph_kmeans

import networkx as nx

G = ox.load_graphml("./graph.graphml")
G = G.to_undirected()
graphs = partition_graph_kmeans(G, 5)


ox.plot_graph(graphs[0])