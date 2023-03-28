# This code will take existing graph structures and seed them with opinions according to the specifications in the code.
# It will then run an opinion dynamic model and save the results as a gml file.


import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as opn
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from Modified_Algorithmic_Bias import *
from Homophily_generated_networks import *
from collections import Counter
plt.matplotlib.use('Qt5Agg')
from tqdm import tqdm


# Network topology
# parameters governing the graph structure

n = 1000 # number of nodes Note: This should be an even number to ensure stability
m = 6 # number of edges per node
p = 0.70 # probability of rewiring each edge
minority_fraction = 0.5 # fraction of minority nodes in the network
similitude = 0.8 # similarity metric



g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

# Model selection
model = AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 1) #bounded confidence parameter
config.add_model_parameter("mu", 0.7) #convergence parameter
config.add_model_parameter("gamma", 0) #bias parameter
config.add_model_parameter("mode", "normal") #initial opinion distribution
config.add_model_parameter("noise", 0.05) # noise parameter that cannot exceed 10%
config.add_model_parameter("minority_fraction", minority_fraction) # minority fraction in the network
model.set_initial_status(config)

# Simulation execution
epochs = 50
iterations = model.iteration_bunch(epochs)

# Iteration extraction
test_vector = iterations[1]['status']
control_graph = g.copy()

# assigning opinions to nodes
for nodes in control_graph.nodes:
     control_graph.nodes[nodes]['opinion'] = test_vector[nodes]

#print("assortivity before opinion dynamics for color", nx.attribute_assortativity_coefficient(control_graph, 'color'))
print("assortivity before opinion dynamics for opinion", nx.numeric_assortativity_coefficient(control_graph, 'opinion'))
opinion_vector = iterations[epochs-1]['status']


for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]

# visualization
x =nx.get_node_attributes(g, 'opinion')
int = list(x.values())
#print("this should be the opinions after x iterations", np.mean(int))

#showing distribution of opinions after opinion dynamics
plt.hist(int, range = (-1,1), bins = 50)
plt.show()
plt.close()

pos = nx.spring_layout(g, k=5, iterations = 10, scale = 10)
nx.draw(g, node_color = int, with_labels = False,
       alpha = 0.6, node_size = 50, vmin = 0, vmax = 1)
#nx.draw(g, node_color = int)
plt.show()


# assortativity after opinion dynamics
print("assortivity after opinion dynamics is:", nx.numeric_assortativity_coefficient(g, 'opinion'))

viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")


# Graveyard

# color_list = nx.get_node_attributes(control_graph, 'color')
# print(Counter(color_list.values()))


# checking if the color and opinion vectors are the allocated properly

# for i in range(len(color_list)):
#     if color_list[i] == 'red' and test_vector[i] == 0:
#         print("error red", i, color_list[i], test_vector[i])
#     if color_list[i] == 'blue' and test_vector[i] == 1:
#         print("error blue", i, color_list[i], test_vector[i])