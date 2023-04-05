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
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#from tkinter import *
import pandas as pd
import seaborn as sns

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
config.add_model_parameter("mode", "mixed") #initial opinion distribution
config.add_model_parameter("noise", 0) # noise parameter that cannot exceed 10%
config.add_model_parameter("minority_fraction", minority_fraction) # minority fraction in the network
config.add_model_parameter("dims", d) # number of dimensions
config.add_model_parameter("gamma_cov", 0.35) # correlation between dimensions
model.set_initial_status(config)


# Defining helper functions to perform seeding

def Extract(lst):
    return [item[0] for item in lst]


#Simulation execution
epochs = 5
iterations = model.iteration_bunch(epochs, node_status = True, progress_bar = True)

test_vector = iterations[0]['status']
control_graph = g.copy()
#print("Initial distribution", test_vector)

# assigning opinions to nodes
for nodes in control_graph.nodes:
     control_graph.nodes[nodes]['opinion'] = test_vector[nodes]

x_before =nx.get_node_attributes(control_graph, 'opinion')
int = list(x_before.values())
data_1 =np.array(int)

#print("assortivity before opinion dynamics for color", nx.attribute_assortativity_coefficient(control_graph, 'color'))
#print("assortivity before opinion dynamics for opinion", nx.numeric_assortativity_coefficient(control_graph, 'opinion'))
opinion_vector = iterations[epochs-1]['status']
#print("Final distribution", opinion_vector)

for nodes in g.nodes:
     g.nodes[nodes]['opinion'] = opinion_vector[nodes]

# visualization


"Visualization before and after "
x_after =nx.get_node_attributes(g, 'opinion')
int = list(x_after.values())
data_2 =np.array(int)

df_before = pd.DataFrame(data_1, columns=[f'Dimension {i + 1}' for i in range(d)])
df_after = pd.DataFrame(data_2, columns=[f'Dimension {i + 1}' for i in range(d)])


# Plot histograms
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

# Plot histograms
for i, col in enumerate(df_before.columns):
    axes[i].hist(df_before[col], bins=20, alpha=0.5, label='Before Opinion Dynamics', color='blue')
    axes[i].hist(df_after[col], bins=20, alpha=0.5, label='After Opinion Dynamics', color='green')
    axes[i].set_title(f'Histogram for {col}')
    axes[i].legend()

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.show()

#trying to plot evolution per dimension
fig, axes = plt.subplots(d, 1, figsize=(6, d * 3), sharex=True)

for d_index in range(d):
    # Extract the dth opinion for all nodes across iterations
    opinions_d = [[opinions[d_index] for node, opinions in iteration['status'].items()] for iteration in iterations]

    # Transpose the opinions_d list for easier plotting
    opinions_d_T = list(zip(*opinions_d))

    # Plot the dth opinion for each node
    for i, node_opinions in enumerate(opinions_d_T):
        axes[d_index].plot(node_opinions, color='black', alpha = 0.5)

    axes[d_index].set_ylabel(f'Opinion {d_index}')
    axes[d_index].legend()

plt.xlabel('Iterations')
fig.tight_layout()
plt.show()




##for d in range(d):
#    opinions_d = [[node[opinions[d]] for node, opinions in iteration.items()] for iteration in iterations]
#    print(opinions_d)

viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")
