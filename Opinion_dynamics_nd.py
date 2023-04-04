#This code will take existing graph structures and seed them with opinions according to the specifications in the code.
#In particular, we will be seeding a graph with multidimensional opions and then running an opinion dynamics model on it.

import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as opn
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from Modified_algorithmic_bias_nd import *
from Homophily_generated_networks import *
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#from tkinter import *


# Network topology
# parameters governing the graph structure

n = 1000 # number of nodes Note: This should be an even number to ensure stability
m = 8 # number of edges per node
p = 0.70 # probability of rewiring each edge
minority_fraction = 0.5 # fraction of minority nodes in the network
similitude = 0.8 # similarity metric
d = 4 # number of dimension
#gamma = 0.5 # correlation between dimensions

# Generating graph
g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

# Model selection
model = AlgorithmicBiasModel_nd(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 1) #bounded confidence parameter
config.add_model_parameter("mu", 0.5) #convergence parameter
config.add_model_parameter("gamma", 0) #bias parameter
config.add_model_parameter("mode", "normal") #initial opinion distribution
config.add_model_parameter("noise", 0) # noise parameter that cannot exceed 10%
config.add_model_parameter("minority_fraction", minority_fraction) # minority fraction in the network
config.add_model_parameter("dims", d) # number of dimensions
config.add_model_parameter("gamma_cov", 0.7) # correlation between dimensions
model.set_initial_status(config)


# Defining helper functions to perform seeding

def Extract(lst):
    return [item[0] for item in lst]


#Simulation execution
epochs = 5
iterations = model.iteration_bunch(epochs, node_status = True, progress_bar = True)

# Iteration extraction
#for x in range(epochs):
#   print(iterations[x])


test_vector = iterations[0]['status']
control_graph = g.copy()
print("Initial distribution", test_vector)

# assigning opinions to nodes
for nodes in control_graph.nodes:
     control_graph.nodes[nodes]['opinion'] = test_vector[nodes]

#print("assortivity before opinion dynamics for color", nx.attribute_assortativity_coefficient(control_graph, 'color'))
#print("assortivity before opinion dynamics for opinion", nx.numeric_assortativity_coefficient(control_graph, 'opinion'))
opinion_vector = iterations[epochs-1]['status']
print("Final distribution", opinion_vector)

for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]

# visualization
x =nx.get_node_attributes(g, 'opinion')
int = list(x.values())
#print("this should be the opinions after x iterations", np.mean(int))




viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")
