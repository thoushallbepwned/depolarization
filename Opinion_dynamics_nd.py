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


# Network topology
# parameters governing the graph structure

n = 600 # number of nodes Note: This should be an even number to ensure stability
m = 6 # number of edges per node
p = 0.70 # probability of rewiring each edge
minority_fraction = 0.5 # fraction of minority nodes in the network
similitude = 0.8 # similarity metric
d = 1 # number of dimension
gamma = 0.5 # correlation between dimensions



g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

# Model selection
model = AlgorithmicBiasModel_nd(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 1) #bounded confidence parameter
config.add_model_parameter("mu", 0.7) #convergence parameter
config.add_model_parameter("gamma", 0) #bias parameter
config.add_model_parameter("mode", "normal") #initial opinion distribution
config.add_model_parameter("noise", 0.05) # noise parameter that cannot exceed 10%
config.add_model_parameter("minority_fraction", minority_fraction) # minority fraction in the network
config.add_model_parameter("dims", d) # number of dimensions
config.add_model_parameter("gamma", gamma) # correlation between dimensions
model.set_initial_status(config)


# Defining helper functions to perform seeding

def Extract(lst):
    return [item[0] for item in lst]

# Characterizing multidimensional opinion distributions functions
# def polarized_distr_nd(G, n, minority_fraction, d, gamma):
#     lower, upper = -1, 1  # lower and upper bounds
#
#     count2 = int(n * minority_fraction)
#     count1 = int(n * (1 - minority_fraction))
#
#     s = np.zeros((n, d))
#     correlation_matrix = np.identity(d) * (1 - gamma) + gamma
#     #print(correlation_matrix)
#
#     if d == 1:
#         mu1, sigma1 = np.random.uniform(low=-0.9, high=-0.1), np.random.uniform(low=0.0675,
#                                                                                 high=0.175)  # mean and standard deviation
#         mu2, sigma2 = np.random.uniform(low=0.1, high=0.9), np.random.uniform(low=0.0675,
#                                                                               high=0.175)  # mean and standard deviation
#
#         cov1 = np.outer(sigma1 * sigma1, correlation_matrix)
#         cov1 = np.reshape(cov1, (d, d))
#
#         cov2 = np.outer(sigma2 * sigma2, correlation_matrix)
#         cov2 = np.reshape(cov2, (d, d))
#
#         X1 = np.random.multivariate_normal(mean=[mu1] * d, cov=cov1, size=count1)
#         X2 = np.random.multivariate_normal(mean=[mu2] * d, cov=cov2, size=count2)
#
#         s1 = np.clip(X1, lower, upper)
#         s2 = np.clip(X2, lower, upper)
#
#         s[:count1, :] = s1
#         s[count1:, :] = s2
#
#     if d > 1:
#
#         for i in range(d):
#             mu1, sigma1 = np.random.uniform(low=-0.9, high=-0.1), np.random.uniform(low=0.0675,
#                                                                                     high=0.175)  # mean and standard deviation
#             mu2, sigma2 = np.random.uniform(low=0.1, high=0.9), np.random.uniform(low=0.0675,
#                                                                                   high=0.175)  # mean and standard deviation
#
#             cov1 = np.outer(sigma1 * sigma1, correlation_matrix)
#             cov1 = np.reshape(cov1, (d, d))
#
#             cov2 = np.outer(sigma2 * sigma2, correlation_matrix)
#             cov2 = np.reshape(cov2, (d, d))
#
#             X1 = np.random.multivariate_normal(mean=[mu1] * d, cov=cov1, size=count1)
#             X2 = np.random.multivariate_normal(mean=[mu2] * d, cov=cov2, size=count2)
#
#             s1 = np.clip(X1, lower, upper)
#             s2 = np.clip(X2, lower, upper)
#
#             s[:count1, :] = s1
#             s[count1:, :] = s2
#
#     return s
#
# def normal_distr_nd(G, n, d, gamma):
#
#     lower, upper = -1, 1  # lower and upper bounds
#     mu = np.random.uniform(low=-0.25, high=0.25, size=d)
#     sigma = np.random.uniform(low=0.1, high=0.25, size = d)  #standard deviation
#     s = np.zeros((n, d))
#     correlation_matrix = np.identity(d) * (1 - gamma) + gamma
#     #print(correlation_matrix)
#
#     if d > 1:
#         "Will need to add a substantial amount of code to determine the level of covariance in the data"
#         #covariance = A = np.random.rand(d, d)
#         #print(covariance.shape)
#         #cov = (1 / d) * A.T @ A
#         #print(cov.shape)
#         #cov = np.outer(sigma, correlation_matrix)
#         #print(cov.shape)
#         s = np.random.multivariate_normal(mu, correlation_matrix, n)
#         s = s/np.max(s)
#
#     if d == 1:
#         sigma = np.random.uniform(low=0.1, high=0.25, size = d)  # mean and standard deviation
#         X = stats.truncnorm(
#             (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#
#         truncacted_array = X.rvs(n)
#         s = np.reshape(truncacted_array, (n, 1))
#
#     return (s)
#
# def mixed_distr_nd(G,n, minority_fraction, d, gamma):
#     print("Applying the mixed sampling regime, the following choices were made:")
#     lower, upper = -1, 1  # lower and upper bounds
#
#     count2 = int(n * minority_fraction)
#     count1 = int(n * (1 - minority_fraction))
#
#     s = np.zeros((n, d))
#     correlation_matrix = np.identity(d) * (1 - gamma) + gamma
#     #print(correlation_matrix)
#     options = ["a","b"]
#     for i in range(d):
#         choices = random.choices(options, weights = [1,1], k = 1)
#         #print("what did we pick?", choices)
#
#         if choices[0] == "a":
#             print("going polarized")
#             s_1  = polarized_distr_nd(G,n, minority_fraction, 1, gamma)
#             plt.hist(s_1, range = (-1,1), bins = 50)
#             plt.show()
#         else:
#             print("going normal")
#             s_1 =   normal_distr_nd(G,n, 1, gamma)
#             plt.hist(s_1, range = (-1,1), bins = 50)
#             plt.show()
#         s[:,i] = s_1[:,0]
#
#     return s


#Simulation execution
epochs = 20
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