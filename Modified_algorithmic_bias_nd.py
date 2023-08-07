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
import torch
from torch_geometric.utils import train_test_split_edges, to_undirected, negative_sampling, from_networkx
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch_geometric.utils as utils

from networkx.drawing.nx_agraph import from_agraph
from netdispatch import AGraph
import seaborn as sns

from scipy.spatial.distance import cosine, euclidean


#modified by Paul Verhagen


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


    def compute_scores(self,x, edge_index):

        embeddings = self.forward(x, edge_index)


        # Get the embeddings for the two nodes
        node1_emb = embeddings[edge_index[0]]
        node2_emb = embeddings[edge_index[1]]

        # Calculate the score as the dot product of the two embeddings
        scores = torch.sum(node1_emb * node2_emb, dim=-1)

        return scores


torch.cuda.empty_cache()
"Loading link predictor model here"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" Loading the link predictor to prepare for incorporation"""
model = GraphSAGE(4, 32).to(device)
model.load_state_dict(
    torch.load("predictors_new/predictor_2000_nodes_sequential_mean_clean.pth"))
model.eval()

class AlgorithmicBiasModel_nd(DiffusionModel):
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
        self.original_graph = None

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
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        """
        super(AlgorithmicBiasModel_nd, self).set_initial_status(configuration)


######## Adding major changes here to the node seeding.

        #Helper functions to seed with opinions

        if self.params['model']['fixed'] == True:

            #seed_list = [42,420, 1337, 299792489, 1442, 100]

            #choice = random.choice(seed_list)
            np.random.seed(42)

        def Extract(lst):
            return [item[0] for item in lst]

        def Match(a, b):
            return [elem for elem in a if elem in b]

        def filter(array1, z):
            return [tuple(elem for i, elem in enumerate(tup) if i in z) for tup in array1]

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
                    #print(cov)
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
            #print("Applying the mixed sampling regime, the following choices were made:")
            lower, upper = -1, 1  # lower and upper bounds

            count2 = int(n * minority_fraction)
            count1 = int(n * (1 - minority_fraction))


            s = np.zeros((n, d))
            correlation_matrix = np.identity(d) * (1 - gamma_cov) + gamma_cov

            options = ["a", "b", "a", "b"]
            for i in range(d):
                #choices = random.choices(options, weights=[1, 1], k=1)
                choices = options[i]

                if choices[0] == "a":
                    #print("going polarized")
                    s_1 = polarized_distr_nd(G, n, minority_fraction, 1, gamma_cov)

                else:
                    #print("going normal")
                    s_1 = normal_distr_nd(G, n, 1, gamma_cov)
                s[:, i] = s_1[:, 0]
            return s



        "Seeding the graph"
        # set node status

        #setting node sequence based on attributes
        #print("what is the type over here?", type(self.graph))
        array = nx.get_node_attributes(self.graph, 'color')
        sorted_array = sorted(array.items(), key=lambda x: x[1])
        index_list = Extract(sorted_array)

        #if self.params['model']['noise'] == 0:
            #print("running in noiseless mode")
        #else:
            #print("running in noisy mode")
        if self.actual_iteration == 0:
            #print("storing the original graph and status")
            self.original_graph = copy.deepcopy(self.graph)
            original_status = copy.deepcopy(self.status)
            initial_status = copy.deepcopy(self.status)
            if self.params['model']['mode'] == 'mixed':
                #print("How often are we entering this per iteration?", self.actual_iteration)
                s = mixed_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['minority_fraction'], self.params['model']['dims'], self.params['model']['gamma_cov'])
                sorted_dist = np.sort(s, axis = 0)
                i = 0
                for node in index_list:
                    entry = sorted_dist[i].flatten()
                    if self.params['model']['dims'] == 1:

                        self.status[node] = float(entry)
                    else:
                        self.status[node] = entry.tolist()
                    #print(type(entry))
                    i += 1
                self.initial_status = self.status.copy()

            if self.params['model']['mode'] == 'normal':
                #print("set to normal mode")
                s = normal_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['dims'], self.params['model']['gamma_cov'])
                #print("this is the shape of s", s, s.shape)

                sorted_dist = np.sort(s, axis = 0)
                i = 0
                for node in index_list:
                    entry = sorted_dist[i].flatten()
                    if self.params['model']['dims'] == 1:

                        self.status[node] = float(entry)
                    else:
                        self.status[node] = entry.tolist()
                    #print(type(entry))
                    i += 1

                self.initial_status = self.status.copy()

            if self.params['model']['mode'] == 'polarized':
                #print("set to polarized mode")

                dist = polarized_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['minority_fraction'],self.params['model']['dims'], self.params['model']['gamma_cov'])
                #dist = [(2*x) -1 for x in dist]
                sorted_dist = np.sort(dist, axis = 0)
                #sorted_dist_round = np.round(sorted_dist)
                i = 0
                for node in index_list:
                    entry = sorted_dist[i].flatten()
                    if self.params['model']['dims'] == 1:

                        self.status[node] = float(entry)
                    else:
                        self.status[node] = entry.tolist()
                    i += 1

                self.initial_status = self.status.copy()

        ### Initialization numpy representation

        max_edgees = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2


        if self.params['model']['dims'] == 1:
            nids = np.array(list(self.status.items()))
            self.ids = nids[:, 0]
        else:
            nids = list(self.status.items())
            self.ids = Extract(nids)
        # self.ids = np.array(len(self.graph.nodes()))

        if max_edgees == self.graph.number_of_edges():
            self.sts = nids[:, 1]

        else:
            "Found the location where a bunch of things need to change"

            for i in self.graph.nodes:
                if self.params['model']['dims'] == 1:
                    i_neigh = list(self.graph.neighbors(i))
                    i_ids = nids[:, 0][i_neigh]
                    i_sts = nids[:, 1][i_neigh]
                    self.node_data[i] = (i_ids, i_sts)
                else:
                    i_neigh = list(self.graph.neighbors(i))
                    i_ids = i_neigh
                    #i_ids = [tup[0] for tup in nids if tup[0] in i_neigh]
                    i_sts = [tup[1] for tup in nids if tup[0] in i_neigh]
                    self.node_data[i] = (i_ids, i_sts)

    # def clean_initial_status(self, valid_status=None):
    #     for n, s in future.utils.iteritems(self.status):
    #         if s > 1 or s < 0:
    #             self.status[n] = 0

    @staticmethod
    def prob(distance, gamma, min_dist):
        if distance < min_dist:
            distance = min_dist
        return np.power(distance, -gamma)

    def pb1(self, statuses, i_status):
        dist = np.abs(statuses - i_status)
        null = np.full(statuses.shape[0], 0.00001)
        "Taking a cheat now by only selecting the first dimension"
        if self.params['model']['dims'] > 1:
            dist = dist[:, 0]
        else:
            dist = dist
        max_base = np.maximum(dist, null)
        dists = max_base ** -self.params['model']['gamma']
        return dists

    def iteration(self, node_status=True):
        """
        Execute a single model iteration
        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        # One iteration changes the opinion of N agent pairs using the following procedure:
        # - first one agent is selected
        # - then a second agent is selected based on a probability that decreases with the distance to the first agent
        # - if the two agents have a distance smaller than epsilon, then they change their status to the average of
        # their previous statuses


        "defining some helper functions"

        def softmax_temperature(x, T):
            x = np.array(x)
            x = x / T
            e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
            return e_x / e_x.sum()

        def link_prediction(self, model, actual_status, break_fraction=0.1):
            """
            This function breaks and re-adds edges in a graph based on node opinions.

            Args:
                self: The class instance
                model: The link prediction model
                actual_status: The actual status of nodes
                break_fraction: The fraction of edges to break (default is 0.1)

            Returns:
                None
            """

            if self.actual_iteration > 0:
                link_predictor = model
                networkx_graph = self.graph.copy()

                for edge in self.graph.edges():
                    networkx_graph.add_edge(*edge)
                for node in self.graph.nodes():
                    opinion = actual_status[node]
                    networkx_graph.add_node(node, opinion=opinion)

                # number of links to break
                break_links = int(break_fraction * float(networkx_graph.number_of_edges()))

                # Converting to Pytorch Geometric format
                data = from_networkx(networkx_graph)
                data.x = data.opinion.to(device)

                # breaking links
                all_edges = list(networkx_graph.edges())
                edges_to_remove = random.sample(all_edges, break_links)
                networkx_graph.remove_edges_from(edges_to_remove)

                data2 = from_networkx(networkx_graph)

                edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph.edges())) / len(
                    set(self.original_graph.edges()))
                print("Edge overlap after breaking:", np.round(edge_overlap, 4), "number of edges",
                      networkx_graph.number_of_edges())

                num_nodes = data.num_nodes
                num_neg_samples = data2.edge_index.shape[
                    1]  # Generate as many negative samples as there are positive edges


                #print("What is data.edge_index?", data2.edge_index.shape, data2.edge_index)

                # # Generate negative edges
                # neg_edge_index = utils.negative_sampling(edge_index=data2.edge_index,
                #                                          num_nodes=num_nodes,
                #                                          num_neg_samples=1*num_neg_samples,
                #                                          method="sparse",
                #                                          force_undirected= True)

                "testing how long it takes to run on the complement"

                complement = nx.complement(networkx_graph)
                #
                neg_edge_index = complement.edges()
                neg_edge_index = np.array(list(neg_edge_index)).T
                neg_edge_index = torch.from_numpy(neg_edge_index).long()
                # #print("how large is this thing?", neg_edge_index.shape)


                #print("testing the neg_edge_index", neg_edge_index.shape, neg_edge_index)

                # Concatenate positive and negative edges
                #edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)


                # Generating embeddings with trained model

                with torch.no_grad():

                    z = model(data.x, data2.edge_index.to(device))

                    #print("what is z exactly?", z.shape)
                    #print("show me z", z)

                    # Calculate the link logits
                    link_logits = torch.cat(
                        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()],
                        dim=0)


                    edge_label_index_unflattened = neg_edge_index.t()

                    #print("this is the edge label index this should be source, dest",edge_label_index_unflattened)

                    # Apply sigmoid function to convert logits to probabilities
                    probabilities = torch.sigmoid(link_logits)

                    #print("what are these probabilities?", probabilities)

                sorted_prob_indices = torch.argsort(probabilities, descending=True)

                sorted_edges = edge_label_index_unflattened[sorted_prob_indices.cpu()]
                #print("What are the sorted edges?", sorted_edges.shape)

                # sorted_edges = sorted_edges
                #
                # "comparing overlap between sorted_edges and the existant edges"
                #
                # edge_list_test = networkx_graph.edges()
                #
                # edge_set1 = set(sorted_edges)
                # edge_set2 = set(edge_list_test)
                #
                # common_edges = edge_set1.intersection(edge_set2)
                # print("the common edges are", len(common_edges))




                new_links_added = 0
                link_index = 0

                while new_links_added < break_links:
                    src, dest = sorted_edges[link_index]

                    if not networkx_graph.has_edge(src.item(), dest.item()):
                        networkx_graph.add_edge(src.item(), dest.item())
                        new_links_added += 1
                    link_index += 1

                # potential_new_edges = [(i, j) for i in range(data.num_nodes) for j in range(i + 1, data.num_nodes)
                #                        if not networkx_graph.has_edge(i, j)]
                #
                # edges_to_add = random.sample(potential_new_edges, break_links)
                # networkx_graph.add_edges_from(edges_to_add)

                edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph.edges())) / len(
                     set(self.original_graph.edges()))
                print("Edge overlap after link prediction:", np.round(edge_overlap,4), "number of edges", networkx_graph.number_of_edges())

                #print(networkx_graph)
                self.graph = networkx_graph


        def removal_protocol(self, model, actual_status, tension_threshold = 1.0, break_fraction = 0.1):
            if self.actual_iteration > 0:
                networkx_graph2 = self.graph.copy()

                for edge in self.graph.edges():
                    networkx_graph2.add_edge(*edge)
                for node in self.graph.nodes():
                    opinion = actual_status[node]
                    networkx_graph2.add_node(node, opinion=opinion)

                # Calculate tension
                for edge in networkx_graph2.edges():
                    diff = [((actual_status[edge[0]][d] + 1) - (actual_status[edge[1]][d] + 1)) ** 2 for d in
                            range(self.params['model']['dims'])]
                    tension = np.sqrt(np.sum(diff))
                    networkx_graph2[edge[0]][edge[1]]['tension'] = tension

                # Get high tension edges
                high_tension_edges = [(node1, node2, attrs['tension']) for node1, node2, attrs in
                                      networkx_graph2.edges(data=True)
                                      if 'tension' in attrs and attrs['tension'] > tension_threshold]
                sorted_ht_edges = sorted(high_tension_edges, key=lambda x: x[2], reverse=True)

                # Break links
                break_links = int(break_fraction * float(networkx_graph2.number_of_edges()))
                broken_links = 0
                for edge in sorted_ht_edges:
                    if broken_links >= break_links:
                        break
                    if networkx_graph2.has_edge(*edge[:2]):
                        networkx_graph2.remove_edge(*edge[:2])
                        broken_links += 1

                # Add links back randomly
                complement = nx.complement(networkx_graph2)
                non_edges = list(complement.edges())
                edges_to_add = random.sample(non_edges, break_links)
                for edge in edges_to_add:
                    networkx_graph2.add_edge(*edge)

                self.graph = networkx_graph2

        actual_status = copy.deepcopy(self.status)

        if self.actual_iteration == 0:
            self.actual_iteration += 1

            delta, node_count, status_delta = self.status_delta(self.status)
            #print(delta, node_count, status_delta)
            if node_status:
                #print("what is the status here?", actual_status)
                return {"iteration": 0, "status": actual_status,
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        n = self.graph.number_of_nodes()


        "Starting the coding block here that allows for link prediction to be on or off"

        "Storing a version of the original graph to calibrate against"


        if self.params['model']['link_prediction'] == "intervened":


            if self.actual_iteration % 4 == 0:
                    link_prediction(self, model, actual_status, break_fraction = 0.1)

            #print("Are we reaching the intervened code at all?")

            # if self.actual_iteration > 0:
            #     link_predictor = model
            #     networkx_graph = self.graph.copy()
            #
            #     for edge in self.graph.edges():
            #         networkx_graph.add_edge(*edge)
            #     for node in self.graph.nodes():
            #         opinion = actual_status[node]
            #         networkx_graph.add_node(node, opinion = opinion)
            #
            #
            #     #number of links to break
            #     break_links = int(0.1*float(networkx_graph.number_of_edges())) #change this later
            #
            #     #Converting to Pytorch Geometric format
            #     data = from_networkx(networkx_graph)
            #     data.x = data.opinion.to(device)
            #
            #     #breaking links
            #     all_edges = list(networkx_graph.edges())
            #     edges_to_remove = random.sample(all_edges, break_links)
            #     networkx_graph.remove_edges_from(edges_to_remove)
            #
            #     data2 = from_networkx(networkx_graph)
            #
            #     edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph.edges())) / len(
            #         set(self.original_graph.edges()))
            #     print("Edge overlap after breaking:", np.round(edge_overlap,4),"number of edges", networkx_graph.number_of_edges())
            #
            #     #print("This is the shape of the full adjacency matrix", full_ad.shape, "average value", np.mean(full_ad))
            #
            #     num_nodes = data.num_nodes
            #     num_neg_samples = data2.edge_index.shape[
            #         1]  # Generate as many negative samples as there are positive edges
            #
            #
            #     #print("What is data.edge_index?", data2.edge_index.shape, data2.edge_index)
            #
            #     # # Generate negative edges
            #     # neg_edge_index = utils.negative_sampling(edge_index=data2.edge_index,
            #     #                                          num_nodes=num_nodes,
            #     #                                          num_neg_samples=1*num_neg_samples,
            #     #                                          method="sparse",
            #     #                                          force_undirected= True)
            #
            #     "testing how long it takes to run on the complement"
            #
            #     complement = nx.complement(networkx_graph)
            #     #
            #     neg_edge_index = complement.edges()
            #     neg_edge_index = np.array(list(neg_edge_index)).T
            #     neg_edge_index = torch.from_numpy(neg_edge_index).long()
            #     # #print("how large is this thing?", neg_edge_index.shape)
            #
            #
            #     #print("testing the neg_edge_index", neg_edge_index.shape, neg_edge_index)
            #
            #     # Concatenate positive and negative edges
            #     #edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
            #
            #
            #     # Generating embeddings with trained model
            #
            #     with torch.no_grad():
            #
            #         z = model(data.x, data2.edge_index.to(device))
            #
            #         #print("what is z exactly?", z.shape)
            #         #print("show me z", z)
            #
            #         # Calculate the link logits
            #         link_logits = torch.cat(
            #             [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()],
            #             dim=0)
            #
            #
            #         edge_label_index_unflattened = neg_edge_index.t()
            #
            #         #print("this is the edge label index this should be source, dest",edge_label_index_unflattened)
            #
            #         # Apply sigmoid function to convert logits to probabilities
            #         probabilities = torch.sigmoid(link_logits)
            #
            #         #print("what are these probabilities?", probabilities)
            #
            #     sorted_prob_indices = torch.argsort(probabilities, descending=True)
            #
            #     sorted_edges = edge_label_index_unflattened[sorted_prob_indices.cpu()]
            #     #print("What are the sorted edges?", sorted_edges.shape)
            #
            #     # sorted_edges = sorted_edges
            #     #
            #     # "comparing overlap between sorted_edges and the existant edges"
            #     #
            #     # edge_list_test = networkx_graph.edges()
            #     #
            #     # edge_set1 = set(sorted_edges)
            #     # edge_set2 = set(edge_list_test)
            #     #
            #     # common_edges = edge_set1.intersection(edge_set2)
            #     # print("the common edges are", len(common_edges))
            #
            #
            #
            #
            #     new_links_added = 0
            #     link_index = 0
            #
            #     while new_links_added < break_links:
            #         src, dest = sorted_edges[link_index]
            #
            #         if not networkx_graph.has_edge(src.item(), dest.item()):
            #             networkx_graph.add_edge(src.item(), dest.item())
            #             new_links_added += 1
            #         link_index += 1
            #
            #     # potential_new_edges = [(i, j) for i in range(data.num_nodes) for j in range(i + 1, data.num_nodes)
            #     #                        if not networkx_graph.has_edge(i, j)]
            #     #
            #     # edges_to_add = random.sample(potential_new_edges, break_links)
            #     # networkx_graph.add_edges_from(edges_to_add)
            #
            #     edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph.edges())) / len(
            #          set(self.original_graph.edges()))
            #     print("Edge overlap after link prediction:", np.round(edge_overlap,4), "number of edges", networkx_graph.number_of_edges())
            #
            #     #print(networkx_graph)
            #     self.graph = networkx_graph

            #print("Completed one loop of link prediction")

            #print(networkx_graph)

        if self.params['model']['link_prediction'] == "natural":
            print(f"We are entering natural loop for iteration {self.actual_iteration}")
            pass

        if self.params['model']['link_prediction'] == "targeted":

            if self.actual_iteration % 4 == 0:
                removal_protocol(self, model, actual_status, tension_threshold=1.0, break_fraction=0.1)

            # if self.params['model']['operation_mode'] == "ensemble":
            #
            # print("entering the targeted code section")
            # "Looking at the edgelist and calculating difference metrics for entire array"
            #
            # if self.actual_iteration > 0:
            #     networkx_graph2 = self.graph.copy()
            #
            #     for edge in self.graph.edges():
            #         networkx_graph2.add_edge(*edge)
            #     for node in self.graph.nodes():
            #         opinion = actual_status[node]
            #         networkx_graph2.add_node(node, opinion = opinion)
            #
            # "introduction notion of tension: mean euclidean difference between two connected node opinions"
            #
            # edge_tension = []
            # for edge in networkx_graph2.edges():
            #     diff = [((actual_status[edge[0]][d] + 1) - (actual_status[edge[1]][d] + 1)) ** 2 for d in
            #             range(self.params['model']['dims'])]
            #     tension = np.sqrt(np.sum(diff))
            #     networkx_graph2[edge[0]][edge[1]]['tension'] = tension
            #
            #
            # # creating a list of tensions that fall above the threshold.
            #
            # tension_threshold = 1.0
            #
            # high_tension_edges = [(node1, node2, attrs['tension']) for node1, node2, attrs in networkx_graph2.edges(data=True)
            #                       if 'tension' in attrs and attrs['tension'] > tension_threshold]
            #
            #
            # sorted_ht_edges = sorted(high_tension_edges, key=lambda x: x[2], reverse=True)
            # print("how many high tension edges are there?", len(high_tension_edges))
            #
            # # number of links to break
            # break_links = int(0.1 * float(networkx_graph2.number_of_edges()))  # change this later
            #
            # broken_links = 0
            # broken_index = 0
            #
            #
            # for edge in sorted_ht_edges:
            #     if broken_links >= break_links:
            #         break
            #     if networkx_graph2.has_edge(*edge[:2]):
            #         networkx_graph2.remove_edge(*edge[:2])
            #
            #         broken_links += 1
            #         #print(broken_links)
            #
            # edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph2.edges())) / len(
            #     set(self.original_graph.edges()))
            # print("Edge overlap after breaking in targeted:", np.round(edge_overlap, 4), "number of edges",
            #       networkx_graph2.number_of_edges())
            #
            #
            #
            # #reintroducing links at random
            #
            # complement = nx.complement(networkx_graph2)
            # non_edges = list(complement.edges())
            #
            # edges_to_add = random.sample(non_edges, break_links)
            #
            # for edge in edges_to_add:
            #     networkx_graph2.add_edge(*edge)
            #
            # edge_overlap = len(set(self.original_graph.edges()).intersection(networkx_graph2.edges())) / len(
            #     set(self.original_graph.edges()))
            # print("Edge overlap after restoration in targeted:", np.round(edge_overlap, 4), "number of edges",
            #       networkx_graph2.number_of_edges())
            #
            # self.graph = networkx_graph2

        else:
            #print("No legal intervention mode selected, exiting", self.params['model']['link_prediction'])
            pass



        #interact with peers
        #print(f"how often are we entering this, checking variables {self.actual_iteration} and {self.params['model']['link_prediction']} and {self.params['model']['epsilon']}")
        for i in range(0, n):

            # Selecting a random node
            # n1 = list(self.graph.nodes)[np.random.randint(0, n)]
            n1 = int(choice(self.ids))

            if len(self.node_data) == 0:
                sts = self.sts
                ids = self.ids
                # Removing the node selected
                # In some cases the node would self-interact with sufficiently high gamma.
                neigh_sts = np.delete(sts, n1)
                neigh_ids = np.delete(ids, n1)
            else:
                neigh_ids = self.node_data[n1][0]
                neigh_sts = np.array([actual_status[id] for id in neigh_ids])

            # selecting the node based on neighbors' status = self.node_data[n1][1]
            # using neighbor_status and actual_status[n1] as parameters
            # If we use self.status[n1] we get the previous iteration but not updated

            # Selecting a random node based on the probability
            #print("currently targeted node is", n1)

            "Putting a lazy fix here: I am flattening the actual_status of [n1] to select the node"

            if self.params['model']['dims'] == 1:
                selection_prob = self.pb1(neigh_sts, actual_status[n1])
            else:
                selection_prob = self.pb1(neigh_sts, actual_status[n1][0])

            total = np.sum(selection_prob)
            selection_prob = selection_prob / total
            cumulative_selection_probability = np.cumsum(selection_prob)

            r = np.random.random_sample()
            # n2 = np.argmax(cumulative_selection_probability >= r) -1
            n2 = np.argmax(
                cumulative_selection_probability >= r)
            # seleziono n2 dagli id dei neighbors di n1
            n2 = int(neigh_ids[n2])



            "Stuff gets interesting here, because we will need to specify a distance metric here to govern the interaction"
            "Adding forloop here that will iterate over dimensions"

            "converting epsilon into a cosine metric"
            cos_epsilon = self.params['model']['epsilon'] * 2 - 1

            "Setting 1: mean euclidean distance"

            "This represents the default mode, where the mean euclidean distance (taken over all dimensions) is used to govern the interaction"

            if self.params['model']['distance_method'] == 'mean_euclidean':
                if self.params['model']['dims'] > 1:
                    diff = [abs((actual_status[n1][d]+2) - (actual_status[n2][d]+2)) for d in range(self.params['model']['dims'])]
                    diff = np.mean(diff)
                else:
                    diff  = np.abs((actual_status[n1]+2) - (actual_status[n2]+2))

                diff = np.array(diff)

            "Setting 2: Strict Euclidean distance"

            "This represents the strict mode, where the maximum euclidean distance (taken over all dimensions) is used to govern the interaction"

            if self.params['model']['distance_method'] == 'strict_euclidean':
                if self.params['model']['dims'] > 1:
                    diff = [abs((actual_status[n1][d]+2) - (actual_status[n2][d]+2)) for d in range(self.params['model']['dims'])]
                    diff = np.max(diff)
                else:
                    diff  = np.abs((actual_status[n1]+2) - (actual_status[n2]+2))

                diff = np.array(diff)

            "Setting 3: Cosine distance"

            "This represents the cosine mode, where the cosine distance (taken over all dimensions) is used to govern the interaction"

            if self.params['model']['distance_method'] == 'cosine':
                if self.params['model']['dims'] == 1:
                    print("Single dimensional cosine distance does not make sense, changing parameters to euclidean")
                    diff = np.abs((actual_status[n1] + 2) - (actual_status[n2] + 2))

                else:

                    diff = np.dot(actual_status[n1], actual_status[n2])/ (np.linalg.norm(actual_status[n1]) * np.linalg.norm(actual_status[n2]))
                    #print(diff)


            "Setting 4: Size cosine distance"

            "This represents the size cosine mode, where the cosine distance (taken over all dimensions) without correcting for size is used to govern the interaction"
            if self.params['model']['distance_method'] == 'size_cosine':
                if self.params['model']['dims'] == 1:
                    print("Single dimensional cosine distance does not make sense, changing parameters to euclidean")
                    diff = np.abs((actual_status[n1] + 2) - (actual_status[n2] + 2))
                else:
                    "Taking the biggest vector"
                    cosine_sim = np.dot(actual_status[n1], actual_status[n2])#/ (np.linalg.norm(actual_status[n1]) * np.linalg.norm(actual_status[n2]))
                    #euclidean_dist = euclidean(actual_status[n1], actual_status[n2])
                    diff = (cosine_sim)
                    #print("this is the diff:", diff)


            ############################################################################################################
            "First model: We are going to iterate over all admissible dimension"



            "creating allowance parameters"

            if self.params['model']['distance_method'] == 'euclidean':
                if diff < self.params['model']['epsilon']:
                    allowance = True
                else:
                    allowance = False
            elif self.params['model']['distance_method'] == 'cosine':
                if diff < cos_epsilon:
                    allowance = True
                else:
                    allowance = False
            elif self.params['model']['distance_method'] == 'size_cosine':
                if diff < cos_epsilon:
                    allowance = True
                else:
                    allowance = False
            elif self.params['model']['distance_method'] == 'mean_euclidean':
                if diff < self.params['model']['epsilon']:
                    allowance = True
                else:
                    allowance = False
            elif self.params['model']['distance_method'] == 'strict_euclidean':
                if diff < self.params['model']['epsilon']:
                    allowance = True
                else:
                    allowance = False


            "Setting the mode of interaction"

            "Mode 1: Interactions are characterized by whole of opinion vector"

            if self.params['model']['operational_mode'] == 'ensemble':

                if self.params['model']['dims'] > 1:
                    #print("This means that we are working in multidimensional space")
                    for dim in range(self.params['model']['dims']):

                        if allowance == True:
                            #print("Allowance is true, we are going to interact")

                            # Adding a little bit of extra noise into the equation
                            if self.params['model']['noise'] > 0:


                                change1 = ((actual_status[n2][dim]+2) - (actual_status[n1][dim]+2))
                                change2 = ((actual_status[n1][dim]+2) - (actual_status[n2][dim]+2))


                                noise1 = np.random.uniform(low=-1*change1*self.params['model']['noise'], high=change1*self.params['model']['noise'])
                                noise2 = np.random.uniform(low=-1*change2*self.params['model']['noise'], high=change2*self.params['model']['noise'])


                                actual_status[n1][dim] = actual_status[n1][dim]+ self.params['model']['mu']*change1 + noise1
                                actual_status[n2][dim] = actual_status[n2][dim]+ self.params['model']['mu']*change2 + noise2

                            if self.params['model']['noise'] == 0:

                                # Testing some ways to see if the absolute difference is screwing things up
                                change1 = (actual_status[n2][dim]+10) - (actual_status[n1][dim]+10)
                                change2 = (actual_status[n1][dim]+10) - (actual_status[n2][dim]+10)
                                pos1 = actual_status[n1][dim]
                                pos2 = actual_status[n2][dim]

                    #########################################
                                #THIS EQUATION IS SUPER IMPORTANT

                                #if actual_status[n1] > actual_status[n2]:
                                actual_status[n1][dim] = pos1 + self.params['model']['mu']*change1
                                actual_status[n2][dim] = pos2 + self.params['model']['mu']*change2


                            if len(self.node_data) == 0:
                                self.sts[n1][dim] = actual_status[n1][dim]
                                self.sts[n2][dim] = actual_status[n2][dim]
                else:
                    #print("This mean we are working in unidimensional space")
                    if diff < self.params['model']['epsilon']:
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2] + 2) - (actual_status[n1] + 2))
                            change2 = ((actual_status[n1] + 2) - (actual_status[n2] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1] = actual_status[n1] + self.params['model']['mu'] * change1 + noise1
                            actual_status[n2] = actual_status[n2] + self.params['model']['mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2] + 10) - (actual_status[n1]+ 10)
                            change2 = (actual_status[n1] + 10) - (actual_status[n2] + 10)
                            pos1 = actual_status[n1]
                            pos2 = actual_status[n2]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1] = actual_status[n1]
                            self.sts[n2] = actual_status[n2]

            "Mode 2: Interactions are characterized by a randomly selected dimension of the opinion vector"

            if self.params['model']['operational_mode'] == 'sequential':
                #print("This means that we are working in iterative mode")
                if self.params['model']['dims'] > 1:
                    # print("Going into softmax operational mode")

                    if allowance == True:
                        # print("Allowance is true, we are going to interact")

                        "Creating a difference array between the two nodes and selecting through softmax based on that difference"
                        #difference_array = np.abs(np.array(actual_status[n1]) - np.array(actual_status[n2]))
                        #probabilities = softmax(difference_array)

                        chosen_dimension = np.random.choice(self.params['model']['dims'])
                        # print("difference_array: ", difference_array)
                        # print("Probabilities: ", probabilities)
                        #print("chosen dimension", chosen_dimension)
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2][chosen_dimension] + 2) - (
                                        actual_status[n1][chosen_dimension] + 2))
                            change2 = ((actual_status[n1][chosen_dimension] + 2) - (
                                        actual_status[n2][chosen_dimension] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1][chosen_dimension] = actual_status[n1][chosen_dimension] + self.params['model'][
                                'mu'] * change1 + noise1
                            actual_status[n2][chosen_dimension] = actual_status[n2][chosen_dimension] + self.params['model'][
                                'mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2][chosen_dimension] + 10) - (
                                        actual_status[n1][chosen_dimension] + 10)
                            change2 = (actual_status[n1][chosen_dimension] + 10) - (
                                        actual_status[n2][chosen_dimension] + 10)
                            pos1 = actual_status[n1][chosen_dimension]
                            pos2 = actual_status[n2][chosen_dimension]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1][chosen_dimension] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2][chosen_dimension] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1][chosen_dimension] = actual_status[n1][chosen_dimension]
                            self.sts[n2][chosen_dimension] = actual_status[n2][chosen_dimension]
                else:
                    #print("This mean we are working in unidimensional space")
                    if diff < self.params['model']['epsilon']:
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2] + 2) - (actual_status[n1] + 2))
                            change2 = ((actual_status[n1] + 2) - (actual_status[n2] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1] = actual_status[n1] + self.params['model']['mu'] * change1 + noise1
                            actual_status[n2] = actual_status[n2] + self.params['model']['mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2] + 10) - (actual_status[n1]+ 10)
                            change2 = (actual_status[n1] + 10) - (actual_status[n2] + 10)
                            pos1 = actual_status[n1]
                            pos2 = actual_status[n2]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1] = actual_status[n1]
                            self.sts[n2] = actual_status[n2]

            "Mode 3: Softmax, interactions are governed proportional to opinion difference"

            if self.params['model']['operational_mode'] == 'softmax':
                if self.params['model']['dims'] > 1:
                    #print("Going into softmax operational mode")

                    if allowance == True:
                        # print("Allowance is true, we are going to interact")

                        "Creating a difference array between the two nodes and selecting through softmax based on that difference"
                        difference_array = np.abs(np.array(actual_status[n1]) - np.array(actual_status[n2]))

                        probabilities = softmax_temperature(-(difference_array - np.min(difference_array)), 0.5)

                        chosen_dimension = np.random.choice(self.params['model']['dims'], p=probabilities)
                        #print("difference_array: ", difference_array)
                        #print("Probabilities: ", probabilities)
                        #print("chosen dimension", chosen_dimension)
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2][chosen_dimension] + 2) - (actual_status[n1][chosen_dimension] + 2))
                            change2 = ((actual_status[n1][chosen_dimension] + 2) - (actual_status[n2][chosen_dimension] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1][chosen_dimension] = actual_status[n1][chosen_dimension] + self.params['model'][
                                'mu'] * change1 + noise1
                            actual_status[n2][chosen_dimension] = actual_status[n2][chosen_dimension] + self.params['model'][
                                'mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2][chosen_dimension] + 10) - (actual_status[n1][chosen_dimension] + 10)
                            change2 = (actual_status[n1][chosen_dimension] + 10) - (actual_status[n2][chosen_dimension] + 10)
                            pos1 = actual_status[n1][chosen_dimension]
                            pos2 = actual_status[n2][chosen_dimension]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1][chosen_dimension] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2][chosen_dimension] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1][chosen_dimension] = actual_status[n1][chosen_dimension]
                            self.sts[n2][chosen_dimension] = actual_status[n2][chosen_dimension]
                else:
                    # print("This mean we are working in unidimensional space")
                    if diff < self.params['model']['epsilon']:
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2] + 2) - (actual_status[n1] + 2))
                            change2 = ((actual_status[n1] + 2) - (actual_status[n2] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1] = actual_status[n1] + self.params['model']['mu'] * change1 + noise1
                            actual_status[n2] = actual_status[n2] + self.params['model']['mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2] + 10) - (actual_status[n1] + 10)
                            change2 = (actual_status[n1] + 10) - (actual_status[n2] + 10)
                            pos1 = actual_status[n1]
                            pos2 = actual_status[n2]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1] = actual_status[n1]
                            self.sts[n2] = actual_status[n2]

            "Mode 4: Bounded, interactions are governed by a bounded difference"

            if self.params['model']['operational_mode'] == 'bounded':
                #print("This means that we are working in bounded mode")
                diff_bounded = self.params['model']['epsilon']

                if self.params['model']['dims'] > 1:
                    # print("Going into softmax operational mode")

                    if allowance == True:
                        # print("Allowance is true, we are going to interact")


                        for i in range(self.params['model']['dims']):
                            if np.abs((actual_status[n1][i]+2) - (actual_status[n2][i]+2)) < diff_bounded:
                                #print("We found a dimension that is close enough to interact on", i)

                                if self.params['model']['noise'] > 0:
                                    change1 = ((actual_status[n2][i] + 2) - (
                                            actual_status[n1][i] + 2))
                                    change2 = ((actual_status[n1][i] + 2) - (
                                            actual_status[n2][i] + 2))

                                    noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                               high=change1 * self.params['model']['noise'])
                                    noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                               high=change2 * self.params['model']['noise'])

                                    actual_status[n1][i] = actual_status[n1][i] + self.params['model'][
                                        'mu'] * change1 + noise1
                                    actual_status[n2][i] = actual_status[n2][i] + self.params['model'][
                                        'mu'] * change2 + noise2

                                if self.params['model']['noise'] == 0:
                                    # Testing some ways to see if the absolute difference is screwing things up
                                    change1 = (actual_status[n2][i] + 10) - (
                                            actual_status[n1][i] + 10)
                                    change2 = (actual_status[n1][i] + 10) - (
                                            actual_status[n2][i] + 10)
                                    pos1 = actual_status[n1][i]
                                    pos2 = actual_status[n2][i]

                                    #########################################
                                    # THIS EQUATION IS SUPER IMPORTANT

                                    # if actual_status[n1] > actual_status[n2]:
                                    actual_status[n1][i] = pos1 + self.params['model']['mu'] * change1
                                    actual_status[n2][i] = pos2 + self.params['model']['mu'] * change2
                            else:
                                continue

                            if len(self.node_data) == 0:
                                self.sts[n1][i] = actual_status[n1][i]
                                self.sts[n2][i] = actual_status[n2][i]
                else:
                    # print("This mean we are working in unidimensional space")
                    if diff < self.params['model']['epsilon']:
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2] + 2) - (actual_status[n1] + 2))
                            change2 = ((actual_status[n1] + 2) - (actual_status[n2] + 2))

                            noise1 = np.random.uniform(low=-1 * change1 * self.params['model']['noise'],
                                                       high=change1 * self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1 * change2 * self.params['model']['noise'],
                                                       high=change2 * self.params['model']['noise'])

                            actual_status[n1] = actual_status[n1] + self.params['model']['mu'] * change1 + noise1
                            actual_status[n2] = actual_status[n2] + self.params['model']['mu'] * change2 + noise2

                        if self.params['model']['noise'] == 0:
                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2] + 10) - (actual_status[n1] + 10)
                            change2 = (actual_status[n1] + 10) - (actual_status[n2] + 10)
                            pos1 = actual_status[n1]
                            pos2 = actual_status[n2]

                            #########################################
                            # THIS EQUATION IS SUPER IMPORTANT

                            # if actual_status[n1] > actual_status[n2]:
                            actual_status[n1] = pos1 + self.params['model']['mu'] * change1
                            actual_status[n2] = pos2 + self.params['model']['mu'] * change2

                        if len(self.node_data) == 0:
                            self.sts[n1] = actual_status[n1]
                            self.sts[n2] = actual_status[n2]


        delta = actual_status
        node_count = {}
        status_delta = {}

        #print("node status", actual_status)

        self.status = actual_status
        self.actual_iteration += 1
        #self.graph = networkx_graph

        #print("Reached end of iteration", self.actual_iteration)

        if node_status:
            #print("what is the iteration number", self.actual_iteration)
            return {"iteration": self.actual_iteration - 1, "status": delta,
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            print("Are we getting stuck here?")
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

