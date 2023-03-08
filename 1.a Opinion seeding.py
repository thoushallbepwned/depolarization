# This code will take existing graph structures and seed them with opinions accordingo the specifications in the code.
# The code will then save the graph structure with the opinions as a gml file.
# The code will then run the model on the graph structure and save the results as a gml file.

# Importing the libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from random import choices
from tqdm import tqdm

# We will need to create several different regimes for opinion seeding
# 1. a normal distribution 1d seeding that is truncated mean 0, sigma 0.15
# 2. a polarized distribution 1d seeding that is set on 0.8 and -0.8

# Start of main code

# reading GML files


# Input options ("normal", "random", "polarized")
def seeding_opinions(regime):
    graph_vector = os.listdir("data/graph_structures")

    for graph in tqdm(graph_vector):
        G = nx.read_gml(f"data/graph_structures/{graph}")
        n = len(G.nodes())

        # flattening opinions

        # s = np.zeros(n)
        # nx.set_node_attributes(G, 0, 'opinion')

        # setting the node opinions based on regime

        if regime == 'random':
            #print("Going down the random route")
            choice = choices([normal_distr(G,n), polarized_distr(G,n)], weights=[0.5, 0.5], k=1)
            s = np.array(choice)
            text = 'random'
        # regime 1: Truncated normal distribution
        elif regime == 'normal':
            #print("Going down the normal route")
            text = 'normal'
            s= normal_distr(G,n)

        # regime 2: polarized distribution
        elif regime == 'polarized':
            #print("Going down the polarized route")
            text = 'polarized'
            s = polarized_distr(G,n)

#        plt.hist(s)
#        plt.show()
        for i in range(n):
             #print(s[i])
             G.nodes[f'{i}']['opinion'] = s[i]

        nx.write_gml(G, f"data/graph_structures/opinion_seeded/{text}/{graph}")

    return ()



def polarized_distr(G,n):
    lower, upper = -1, 1  # lower and upper bounds
    mu1, sigma1 = np.random.uniform(low=-1, high=-0.25), np.random.uniform(low=0.05,
                                                                           high=0.25)  # mean and standard deviation # mean and standard deviation
    mu2, sigma2 = np.random.uniform(low=0.25, high=1), np.random.uniform(low=0.05,
                                                                         high=0.25)  # mean and standard deviation # mean and standard deviation

    X1 = stats.truncnorm(
        (lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
    X2 = stats.truncnorm(
        (lower - mu2) / sigma2, (upper - mu2) / sigma2, loc=mu2, scale=sigma2)

    count = int(n / 2)
    s1 = X1.rvs(count)
    s2 = X2.rvs(count)

    s = np.append(s1, s2)
    return (s)

def normal_distr(G,n):
    lower, upper = -1, 1  # lower and upper bounds
    mu, sigma = np.random.uniform(low=-0.25, high=0.25), np.random.uniform(low=0.05,
                                                                           high=0.25)  # mean and standard deviation

    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    s = X.rvs(n)

    text = 'normal'
    return (s)



seeding_opinions("normal")
#opinion_dict = {v: k for v, k in enumerate(s)}
#print(opinion_dict)
#nx.set_node_attributes(G, opinion_dict, name= 'opinion')
