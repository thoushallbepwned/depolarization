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

n = 1000 # number of nodes
m = 10 # number of edges per node
p = 0.15 # probability of rewiring each edge

g = nx.powerlaw_cluster_graph(n, m, p) # generating Graph


# Model selection
model = AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 0.30)
config.add_model_parameter("gamma", 0)
config.add_model_parameter("mode", "normal")
config.add_model_parameter("noise", 0.025) # noise parameter that cannot exceed 10%
model.set_initial_status(config)


#prior  = nx.numeric_assortativity_coefficient(g, "opinion")
#print("assortivity before opinion dynamics is:", prior)

# Simulation execution
epochs = 20

iterations = model.iteration_bunch(epochs)

# Iteration extraction
opinion_vector = iterations[epochs-1]['status']
#nx.set_node_attributes(g, iterations[3], 'status')
for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]

# visualization
x =nx.get_node_attributes(g, 'opinion')
int = list(x.values())
print("this should be the opinions after x iterations", type(int), np.mean(int))

#showing distribution of opinions after opinion dynamics
plt.hist(int, range = (0,1), bins = 50)
plt.show()

post = nx.numeric_assortativity_coefficient(g, 'opinion') # assortativity after opinion dynamics
print("assortivity after opinion dynamics is:", post)

viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")
