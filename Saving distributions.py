import numpy as np
import os
import pickle
from ndlib.models.DiffusionModel import DiffusionModel
import numpy as np
import random
from random import choice
import future.utils
from collections import defaultdict
import tqdm
import scipy.stats as stats
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import copy
import mpl_toolkits.mplot3d as Axes3D
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
import warnings
import os


from scipy.spatial.distance import cosine, euclidean

n = 1000  # number of nodes Note: This should be an even number to ensure stability
m = 8  # number of edges per node
p = 0.70  # probability of rewiring each edge
minority_fraction = 0.5  # fraction of minority nodes in the network
similitude = 0.8  # similarity metric
d = 4  # number of dimension
# gamma = 0.5 # correlation between dimensions

# Generating graph
g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p)  # generating Graph

# Model selection
model = AlgorithmicBiasModel_nd(g)

"These variables are editable for each experimental run"
# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", epsilon)  # bounded confidence parameter
config.add_model_parameter("mu", 0.5)  # convergence parameter
config.add_model_parameter("gamma", 0)  # bias parameter
config.add_model_parameter("mode", mode)  # initial opinion distribution
config.add_model_parameter("noise", 0)  # noise parameter that cannot exceed 10%
config.add_model_parameter("minority_fraction", minority_fraction)  # minority fraction in the network
config.add_model_parameter("dims", d)  # number of dimensions
config.add_model_parameter("gamma_cov", 0.35)  # correlation between dimensions
config.add_model_parameter("distance_method", distance_method)  # fraction of minority nodes in the network
config.add_model_parameter("fixed", True)  # distribution opinion parameter
model.set_initial_status(config)


class seeding_opinions(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param epsilon: bounded confidence threshold from the Deffuant model, in [0,1]
    :param gamma: strength of the algorithmic bias, positive, real
    Node states are continuous values in [0,1]. How can I change this to become [-1,1]
    The initial state is generated randomly uniformly from the domain [0,1].
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor
             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)

        self.discrete_state = False

        self.available_statuses = {
            "Infected": 0
        }

        self.parameters = {
            "model": {
                "epsilon": {
                    "descr": "Bounded confidence threshold",
                    "range": [0, 2],
                    "optional": False
                },

                "mode": {
                    "descr": "Initialization mode",
                    "range": ["normal", "polarized", "mixed", "none"],
                    "optional": True
                },
                "noise": {
                    "descr": "Noise",
                    "range": [0, 0.1],
                    "optional": True
                },
                "minority_fraction": {
                    "descr": "Minority fraction",
                    "range": [0, 1],
                    "optional": False
                },
                "mu": {
                    "descr": "Convergence parameter",
                    "range": [0, 1],
                    "optional": False
                },
                "gamma": {
                    "descr": "Algorithmic bias",
                    "range": [0, 100],
                    "optional": False
                },
                "gamma_cov": {
                    "descr": "Correlation between dimensions",
                    "range": [0, 1],
                    "optional": False
                },
                "distance_method": {
                    "descr": "Distance method",
                    "range": ["euclidean", "manhattan", "chebyshev"],
                    "optional": False
                },
            },
            "nodes": {},
            "edges": {}
        }

        self.name = "Agorithmic Bias"

        self.node_data = {}
        self.ids = None
        self.sts = None

    def set_initial_status(self, configuration=None):

        def Extract(lst):
            return [item[0] for item in lst]

        def polarized_distr_nd(G, n, minority_fraction, d, gamma_cov):
            lower, upper = -1, 1  # lower and upper bounds

            count2 = int(n * minority_fraction)
            count1 = int(n * (1 - minority_fraction))

            s = np.zeros((n, d))
            correlation_matrix = np.identity(d) * (1 - gamma_cov) + gamma_cov
            # print(correlation_matrix)

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
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # img = ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 3], cmap=plt.hot())
            # fig.colorbar(img)
            # plt.show()


            return s

        def normal_distr_nd(G, n, d, gamma_cov):
            lower, upper = -1, 1  # lower and upper bounds
            s = np.zeros((n, d))

            correlation_matrix = np.identity(d) * (1 - gamma_cov) + gamma_cov
            # if gamma < 0:
            #     correlation_matrix = np.identity(d) * (1 + gamma) - gamma
            # #print(correlation_matrix)

            if d > 1:
                mu = np.random.uniform(low=-0.25, high=0.25, size=d)
                for i in range(d):
                    sigma = np.random.uniform(low=0.1, high=0.25, size=1)  # standard deviation
                    "Will need to add a substantial amount of code to determine the level of covariance in the data"
                    # covariance = A = np.random.rand(d, d)
                    # print(covariance.shape)
                    # cov = (1 / d) * A.T @ A
                    # print(cov.shape)
                    cov = np.outer(sigma * sigma, correlation_matrix)
                    cov = np.reshape(cov, (d, d))
                    # print(cov)
                    s = np.random.multivariate_normal(mu, cov, n)
                    s = s / np.max(s)

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # img = ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 3], cmap=plt.hot())
                    # fig.colorbar(img)
                    # plt.show()

            if d == 1:
                mu = np.random.uniform(low=-0.25, high=0.25, size=d)
                sigma = np.random.uniform(low=0.1, high=0.25, size=d)  # mean and standard deviation
                X = stats.truncnorm(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

                truncacted_array = X.rvs(n)
                s = np.reshape(truncacted_array, (n, 1))

            return (s)

        def mixed_distr_nd(G, n, minority_fraction, d, gamma_cov):
            print("Applying the mixed sampling regime, the following choices were made:")
            lower, upper = -1, 1  # lower and upper bounds

            count2 = int(n * minority_fraction)
            count1 = int(n * (1 - minority_fraction))

            s = np.zeros((n, d))
            correlation_matrix = np.identity(d) * (1 - gamma_cov) + gamma_cov

            options = ["a", "b"]
            for i in range(d):
                choices = random.choices(options, weights=[1, 1], k=1)

                if choices[0] == "a":
                    print("going polarized")
                    s_1 = polarized_distr_nd(G, n, minority_fraction, 1, gamma_cov)

                else:
                    print("going normal")
                    s_1 = normal_distr_nd(G, n, 1, gamma_cov)
                s[:, i] = s_1[:, 0]
            return s

        def save_distributions(mode, filename, dims, graph, minority_fraction, gamma_cov):
            distributions = []
            for _ in range(dims):
                if mode == 'normal':
                    s = normal_distr_nd(graph, len(graph.nodes()), 1, gamma_cov)
                elif mode == "polarized":
                    s = polarized_distr_nd(graph, len(graph.nodes()), minority_fraction, 1, gamma_cov)
                elif mode == "mixed":
                    s = mixed_distr_nd(graph, len(graph.nodes()), minority_fraction, 1, gamma_cov)
                distributions.append(s)
            with open(filename, 'wb') as f:
                pickle.dump(distributions, f)

        # set node status

        # setting node sequence based on attributes
        array = nx.get_node_attributes(self.graph, 'color')
        sorted_array = sorted(array.items(), key=lambda x: x[1])
        index_list = Extract(sorted_array)

        if self.params['model']['noise'] == 0:
            print("running in noiseless mode")
        else:
            print("running in noisy mode")

        # if self.params['model']['mode'] == 'mixed':
        #     print("set to mixed mode")
        #     s = mixed_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['minority_fraction'],
        #                        self.params['model']['dims'], self.params['model']['gamma_cov'])
        #     sorted_dist = np.sort(s, axis=0)
        #     i = 0
        #     for node in index_list:
        #         entry = sorted_dist[i].flatten()
        #         if self.params['model']['dims'] == 1:
        #
        #             self.status[node] = float(entry)
        #         else:
        #             self.status[node] = entry.tolist()
        #         # print(type(entry))
        #         i += 1
        #     self.initial_status = self.status.copy()

        if self.params['model']['mode'] == 'normal':
            print("set to normal mode")


            s = save_distributions(self.params['model']['mode'], self.params['model']['filename'], self.params['model']['dims'], self.graph, self.params['model']['minority_fraction'], self.params['model']['gamma_cov'])
            #s = normal_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['dims'],
                               # self.params['model']['gamma_cov'])
            # print("this is the shape of s", s, s.shape)

            sorted_dist = np.sort(s, axis=0)
            i = 0
            for node in index_list:
                entry = sorted_dist[i].flatten()
                if self.params['model']['dims'] == 1:

                    self.status[node] = float(entry)
                else:
                    self.status[node] = entry.tolist()
                # print(type(entry))
                i += 1

            self.initial_status = self.status.copy()

        if self.params['model']['mode'] == 'polarized':
            print("set to polarized mode")

            dist = polarized_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['minority_fraction'],
                                      self.params['model']['dims'], self.params['model']['gamma_cov'])
            # dist = [(2*x) -1 for x in dist]
            sorted_dist = np.sort(dist, axis=0)
            save_distributions(f"polarized_distributions_{self.params['model']['dims']}d_{self.params['model']['gamma_cov']}gamma_cov.pkl", self.params['model']['dims'], self.graph, self.params['model']['minority_fraction'], self.params['model']['gamma_cov'])
            # sorted_dist_round = np.round(sorted_dist)
            i = 0
            for node in index_list:
                entry = sorted_dist[i].flatten()
                if self.params['model']['dims'] == 1:

                    self.status[node] = float(entry)
                else:
                    self.status[node] = entry.tolist()
                i += 1

            self.initial_status = self.status.copy()


        # def load_distributions(filename):
        #     with open(filename, 'rb') as f:
        #         distributions = pickle.load(f)
        #     return distributions

        # Save the distributions to a file (do this once)
        filename = 'distributions.pkl'
        if not os.path.exists(filename):
            save_distributions(filename, self.params['model']['dims'], self.graph,
                               self.params['model']['minority_fraction'], self.params['model']['gamma_cov'])

        return




        # # Load the distributions during each run
        # if self.params['model']['mode'] == 'mixed':
        #     print("set to mixed mode")
        #     distributions = load_distributions(filename)
        #     s = np.hstack(distributions)
        #     sorted_dist = np.sort(s, axis=0)
        #     i = 0
        #     for node in index_list:
        #         entry = sorted_dist[i].flatten()
        #         if self.params['model']['dims'] == 1:
        #             self.status[node] = float(entry)
        #         else:
        #             self.status[node] = entry.tolist()
        #         i += 1
        #     self.initial_status = self.status.copy()


if __name__ == "__main__":