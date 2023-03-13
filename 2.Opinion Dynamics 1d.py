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
#plt.matplotlib.use('Qt5Agg')


# Network topology

# parameters governing the graph structure

n = 100 # number of nodes
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

prior = nx.attribute_assortativity_coefficient(g, 'opinion')
print("assortivity before opinion dynamics is:", prior)
# Simulation execution
epochs = 100

iterations = model.iteration_bunch(epochs)

opinion_vector = iterations[epochs-1]['status']
initial_distribution = iterations[1]['status']

for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]

# Comparing the initial distribution of opinions with the final distribution of opinions

int_dist = list(initial_distribution.values()) # this is the initial distribution of opinions
end_dist =nx.get_node_attributes(g, 'opinion') # this is the final distribution of opinions
end_int = list(end_dist.values())

print("this should be the opinions after x iterations", len(end_int), np.mean(end_int))

# visualizing the initial and final distributions of opinions
plt.hist(int_dist)
plt.show()

plt.hist(end_int)
plt.show()

# calculating the assortativity of the graph after the opinion dynamics
post = nx.attribute_assortativity_coefficient(g, 'opinion')
print("assortivity after opinion dynamics is:", post)

# printing pdf of opinion dynamics
viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")
