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
from Homophily_generated_networks import *
import mpl_toolkits.mplot3d as Axes3D

# We will need to create several different regimes for opinion seeding
# 1. a normal distribution 1d seeding that is truncated mean 0, sigma 0.15
# 2. a polarized distribution 1d seeding that is set on 0.8 and -0.8

# Start of main code

# reading GML files


# Input options ("normal", "random", "polarized")
def seeding_opinions(n, m, p, minority_faction, similitude, d, gamma, regime):
    n = 1000  # number of nodes Note: This should be an even number to ensure stability
    m = 6  # number of edges per node
    p = 0.70  # probability of rewiring each edge
    minority_fraction = 0.5  # fraction of minority nodes in the network
    similitude = 0.8  # similarity metric
    d = 4 # number of dimension.7
    gamma = 0.5 # correlation between dimensions

    G = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p)  # generating Graph


        # regime 1: Truncated normal distribution
    if regime == 'normal':
        print("Going down the normal route")
        text = 'normal'
        s= normal_distr_nd(G,n, d, gamma)

        # regime 2: polarized distribution
    elif regime == 'polarized':
        print("Going down the polarized route")
        text = 'polarized'
        s = polarized_distr_nd(G,n, minority_fraction, d, gamma)


    elif regime == "mixed":
        s = mixed_distr_nd(G,n, minority_fraction, d, gamma)

    # for i in range(n):
    #      #print(s[i])
    #      G.nodes[f'{i}']['status'] = s[i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(s[:, 0],s[:, 1],s[:, 2], c= s[:,3], cmap = plt.hot())
    fig.colorbar(img)
    plt.show()

   # nx.write_gml(G, f"data/graph_structures/opinion_seeded/{text}/{graph}")

    return ()



def polarized_distr_nd(G, n, minority_fraction, d, gamma):
    lower, upper = -1, 1  # lower and upper bounds

    count2 = int(n * minority_fraction)
    count1 = int(n * (1 - minority_fraction))

    s = np.zeros((n, d))
    correlation_matrix = np.identity(d) * (1 - gamma) + gamma
    #print(correlation_matrix)

    if d == 1:
        mu1, sigma1 = np.random.uniform(low=-0.9, high=-0.1), np.random.uniform(low=0.0675,
                                                                                high=0.175)  # mean and standard deviation
        mu2, sigma2 = np.random.uniform(low=0.1, high=0.9), np.random.uniform(low=0.0675,
                                                                              high=0.175)  # mean and standard deviation

        cov1 = np.outer(sigma1 * sigma1, correlation_matrix)
        cov1 = np.reshape(cov1, (d, d))

        cov2 = np.outer(sigma2 * sigma2, correlation_matrix)
        cov2 = np.reshape(cov2, (d, d))

        X1 = np.random.multivariate_normal(mean=[mu1] * d, cov=cov1, size=count1)
        X2 = np.random.multivariate_normal(mean=[mu2] * d, cov=cov2, size=count2)

        s1 = np.clip(X1, lower, upper)
        s2 = np.clip(X2, lower, upper)

        s[:count1, :] = s1
        s[count1:, :] = s2

    if d > 1:

        for i in range(d):
            mu1, sigma1 = np.random.uniform(low=-0.9, high=-0.1), np.random.uniform(low=0.0675,
                                                                                    high=0.175)  # mean and standard deviation
            mu2, sigma2 = np.random.uniform(low=0.1, high=0.9), np.random.uniform(low=0.0675,
                                                                                  high=0.175)  # mean and standard deviation

            cov1 = np.outer(sigma1 * sigma1, correlation_matrix)
            cov1 = np.reshape(cov1, (d, d))

            cov2 = np.outer(sigma2 * sigma2, correlation_matrix)
            cov2 = np.reshape(cov2, (d, d))

            X1 = np.random.multivariate_normal(mean=[mu1] * d, cov=cov1, size=count1)
            X2 = np.random.multivariate_normal(mean=[mu2] * d, cov=cov2, size=count2)

            s1 = np.clip(X1, lower, upper)
            s2 = np.clip(X2, lower, upper)

            s[:count1, :] = s1
            s[count1:, :] = s2

    return s

def normal_distr_nd(G, n, d, gamma):

    lower, upper = -1, 1  # lower and upper bounds
    mu = np.random.uniform(low=-0.25, high=0.25, size=d)
    sigma = np.random.uniform(low=0.1, high=0.25, size = d)  #standard deviation
    s = np.zeros((n, d))
    correlation_matrix = np.identity(d) * (1 - gamma) + gamma
    print(correlation_matrix)

    if d > 1:
        "Will need to add a substantial amount of code to determine the level of covariance in the data"
        #covariance = A = np.random.rand(d, d)
        #print(covariance.shape)
        #cov = (1 / d) * A.T @ A
        #print(cov.shape)
        #cov = np.outer(sigma, correlation_matrix)
        #print(cov.shape)
        s = np.random.multivariate_normal(mu, correlation_matrix, n)
        s = s/np.max(s)

    if d == 1:
        sigma = np.random.uniform(low=0.1, high=0.25, size = d)  # mean and standard deviation
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        truncacted_array = X.rvs(n)
        s = np.reshape(truncacted_array, (n, 1))

    return (s)

def mixed_distr_nd(G,n, minority_fraction, d, gamma):
    print("Applying the mixed sampling regime, the following choices were made:")
    lower, upper = -1, 1  # lower and upper bounds

    count2 = int(n * minority_fraction)
    count1 = int(n * (1 - minority_fraction))

    s = np.zeros((n, d))
    correlation_matrix = np.identity(d) * (1 - gamma) + gamma
    #print(correlation_matrix)
    options = ["a","b"]
    for i in range(d):
        choices = random.choices(options, weights = [1,1], k = 1)
        #print("what did we pick?", choices)

        if choices[0] == "a":
            print("going polarized")
            s_1  = polarized_distr_nd(G,n, minority_fraction, 1, gamma)
            plt.hist(s_1, range = (-1,1), bins = 50)
            plt.show()
        else:
            print("going normal")
            s_1 =   normal_distr_nd(G,n, 1, gamma)
            plt.hist(s_1, range = (-1,1), bins = 50)
            plt.show()
        s[:,i] = s_1[:,0]

    return s






#seeding_opinions("mixed")
#seeding_opinions("polarized")
seeding_opinions("normal")
#opinion_dict = {v: k for v, k in enumerate(s)}
#print(opinion_dict)
#nx.set_node_attributes(G, opinion_dict, name= 'opinion')
