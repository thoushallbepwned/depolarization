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
g = nx.erdos_renyi_graph(1000, 0.1)
print(g)
#nx.set_node_attributes(g, ('opinion'+1)/2, 'opinion')


#g = nx.read_gml("data/graph_structures/opinion_seeded/normal/graph_0.gml")

# n=1000
# lower, upper = -1, 1  # lower and upper bounds
# mu, sigma = np.random.uniform(low=0, high=1), np.random.uniform(low=0.05,high=0.25)  # mean and standard deviation
#
# X = stats.truncnorm(
#     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#
# s = X.rvs(n)
#
# for i in range(n):
#     # print(s[i])
#     g.nodes[f'{i}']['opinion'] = s[i]
#
# print(g)

# Model selection
model = AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 0.0)
config.add_model_parameter("gamma", 0)
config.add_model_parameter("mode", "normal")
model.set_initial_status(config)

# Simulation execution
iterations = model.iteration_bunch(4)

print(len(iterations))
print((iterations[3]))


print(((iterations[1]['status'])))

opinion_vector = iterations[1]['status']
#nx.set_node_attributes(g, iterations[3], 'status')
for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]
# visualization

x =nx.get_node_attributes(g, 'opinion')

int = list(x.values())
#opinions = np.array(int)
#print(type(opinions))
#print(opinions)
plt.hist(int)
plt.show()



viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")