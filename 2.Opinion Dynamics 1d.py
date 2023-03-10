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
plt.matplotlib.use('Qt5Agg')


# Network topology

# parameters governing the graph structure

n = 10000 # number of nodes
m = 10 # number of edges per node
p = 0.15 # probability of rewiring each edge


g = nx.powerlaw_cluster_graph(n, m, p) # generating Graph

# Model selection
model = AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 0.35)
config.add_model_parameter("gamma", 0)
config.add_model_parameter("mode", "polarized")
model.set_initial_status(config)

# Simulation execution
epochs = 10

iterations = model.iteration_bunch(epochs)

# print(len(iterations))
# print((iterations[3]))
#
#
# print(((iterations[1]['status'])))
print((iterations[1]['status']))
print((iterations[epochs-1]['status']))


opinion_vector = iterations[epochs-1]['status']
#nx.set_node_attributes(g, iterations[3], 'status')
for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]
# visualization

x =nx.get_node_attributes(g, 'opinion')

int = list(x.values())
#opinions = np.array(int)
#print(type(opinions))
print("this should be the opinions after x iterations", len(int), np.mean(int))
plt.hist(int)
plt.show()



viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")
