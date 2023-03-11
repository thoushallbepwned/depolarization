#this code make use of the block models provided in networkx to generate a graph that can later be seeded with opinions


# Importing the libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Start of main code
# Creating the graph

# defining function to create a graph
def create_graph(n, m, p, number):
     # setting parameters for the graph
     #n = 20000 # number of nodes
     #m = 10 # number of edges per node
     #p = 0.15 # probability of rewiring each edge

     for i in tqdm(range(number)):




          G_fb = nx.powerlaw_cluster_graph(n, m, p)

          #setting node properties
          #setting parameters for the opinios

          # mu, sigma = 0, 0.15 # mean and standard deviation
          # s = np.random.normal(mu, sigma, n)
          # #setting the opinion of each node
          # for node in G_fb.nodes():
          #      G_fb.nodes[node]['opinion'] = s[node]
          #      #print(s[node])

          nx.write_gml(G_fb, f"data/graph_structures/graph_{i}.gml")

     return()


create_graph(1000, 10, 0.15, 10)