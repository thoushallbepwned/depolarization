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
plt.matplotlib.use('Qt5Agg')


# Network topology
# parameters governing the graph structure

n = 200 # number of nodes
m = 6 # number of edges per node
p = 0.70 # probability of rewiring each edge
minority_fraction = 0.35 # fraction of minority nodes in the network
similitude = 0.8 # similarity metric



g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

# Model selection
model = AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 0.35)
config.add_model_parameter("gamma", 0)
config.add_model_parameter("mode", "polarized")
config.add_model_parameter("noise", 0.025) # noise parameter that cannot exceed 10%
model.set_initial_status(config)


#prior  = nx.numeric_assortativity_coefficient(g, "opinion")
#print("assortivity before opinion dynamics is:", prior)

# Simulation execution
epochs = 20

iterations = model.iteration_bunch(epochs)

# Iteration extraction
test_vector = iterations[1]['status']
for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = test_vector[nodes]
print("assortivity after opinion dynamics is:", nx.numeric_assortativity_coefficient(g, 'opinion'))
print("assortivity after opinion dynamics for color", nx.attribute_assortativity_coefficient(g, 'color'))

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

print(type(x))

pos = nx.spring_layout(g, k=5, iterations = 10, scale = 10)
nx.draw(g, pos, node_color = int, with_labels = False,
        alpha = 0.6, node_size = 50, vmin = 0, vmax = 1)
#nx.draw(g, node_color = int)
plt.show()


# assortativity after opinion dynamics
print("assortivity after opinion dynamics is:", nx.numeric_assortativity_coefficient(g, 'opinion'))
print("assortivity after opinion dynamics for color", nx.attribute_assortativity_coefficient(g, 'color'))

viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")