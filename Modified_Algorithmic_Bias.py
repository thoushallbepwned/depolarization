from ndlib.models.DiffusionModel import DiffusionModel
import numpy as np
import random
from random import choice
import future.utils
from collections import defaultdict
import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx


__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = ["alina.sirbu@unipi.it", "giulio.rossetti@isti.cnr.it", "valentina.pansanella@sns.it"]
#modified by Paul Verhagen


class AlgorithmicBiasModel(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param epsilon: bounded confidence threshold from the Deffuant model, in [0,1]
    :param gamma: strength of the algorithmic bias, positive, real
    Node states are continuous values in [0,1].
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
                    "range": [0, 1],
                    "optional": False
                },
                "gamma": {
                    "descr": "Algorithmic bias",
                    "range": [0, 100],
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
        super(AlgorithmicBiasModel, self).set_initial_status(configuration)


######## Adding major changes here to the node seeding.

        #Helper functions to seed with opinions
        def polarized_distr(G, n):
            lower, upper = 0, 1  # lower and upper bounds
            mu1, sigma1 = np.random.uniform(low=0, high=0.25), np.random.uniform(low=0.15,
                                                                                   high=0.25)  # mean and standard deviation # mean and standard deviation
            mu2, sigma2 = np.random.uniform(low=0.75, high=1), np.random.uniform(low=0.15,
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

        def normal_distr(G, n):
            lower, upper = 0, 1  # lower and upper bounds
            mu, sigma = np.random.uniform(low=0.25, high=0.75), np.random.uniform(low=0.05,
                                                                                   high=0.125)  # mean and standard deviation

            X = stats.truncnorm(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

            s = X.rvs(n)

            return (s)

        # set node status

        if self.params['model']['mode'] == 'none':
            x = nx.get_node_attributes(self.graph, 'status')

            opinions = list(x.values())
            print(int)
            for node in self.status:
                self.status[node] = opinions[int(node)]
                print(self.status[node])
            print("nice, we can skip this")

        if self.params['model']['mode'] == 'normal':
            print("set to normal mode")
            dist = normal_distr(self.graph, len(self.graph.nodes()))
            plt.hist(dist, range = (0,1), bins = 50)
            plt.show()
            for node in self.status:
                #print(node)
                self.status[node] = dist[node]
            self.initial_status = self.status.copy()

        if self.params['model']['mode']== 'polarized':
            print("set to polarized mode")
            dist = polarized_distr(self.graph, len(self.graph.nodes()))
            plt.hist(dist, range = (0,1), bins = 50)
            plt.show()
            for node in self.status:
                #print(node)
                self.status[node] = dist[node]
            self.initial_status = self.status.copy()

        ### Initialization numpy representation

        max_edgees = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2
        nids = np.array(list(self.status.items()))
        self.ids = nids[:, 0]

        if max_edgees == self.graph.number_of_edges():
            self.sts = nids[:, 1]

        else:
            for i in self.graph.nodes:
                i_neigh = list(self.graph.neighbors(i))
                i_ids = nids[:, 0][i_neigh]
                i_sts = nids[:, 1][i_neigh]
                # non uso mai node_data[:,1]
                # per tenere aggiornato node_data() all'interno del for dovrei ciclare ogni item=nodo
                # e se uno dei suoi vicini è n1 o n2 aggiornare l'array sts
                # la complessità dovrebbe essere O(N)
                # se invece uso solo actual_status, considerando che per ogni nodo ho la lista dei neighbors in memoria
                # a ogni ciclo devo soltanto tirarmi fuori la lista degli stati degli avg_k vicini e prendere i
                # loro stati da actual_status
                # quindi la complessità dovrebbe essere O(N*p) < O(N)
                # sto pensando ad un modo per farlo in O(1) ma non mi è ancora venuto in mente

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

        actual_status = self.status.copy()

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": actual_status,
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        n = self.graph.number_of_nodes()

        # interact with peers
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
            selection_prob = self.pb1(neigh_sts, actual_status[n1])

            # compute probabilities to select a second node among the neighbours
            total = np.sum(selection_prob)
            selection_prob = selection_prob / total
            cumulative_selection_probability = np.cumsum(selection_prob)

            r = np.random.random_sample()
            # n2 = np.argmax(cumulative_selection_probability >= r) -1
            n2 = np.argmax(
                cumulative_selection_probability >= r)
            # seleziono n2 dagli id dei neighbors di n1
            n2 = int(neigh_ids[n2])

            # update status of n1 and n2
            diff = np.abs(actual_status[n1] - actual_status[n2])

            # Testing whether epsilon is respected
            #if diff > self.params['model']['epsilon']:
                #print("ERROR: Node selection opinion out of bounds")

            if diff < self.params['model']['epsilon']:

                # print("Node selection within epsilon bounds")
                # Adding a little bit of extra noise into the equation
                if self.params['model']['noise'] > 0:
                    avg = (actual_status[n1] + actual_status[n2]) / 2.0

                    noise1 = np.random.uniform(low=0, high=avg*self.params['model']['noise'])
                    noise2 = np.random.uniform(low=0, high=avg*self.params['model']['noise'])
                    random_multiplier1 = random.randint(0, 1) *2 -1
                    random_multiplier2 = random.randint(0, 1)*2 - 1


                    actual_status[n1] = avg + noise1*random_multiplier1
                    actual_status[n2] = avg + noise2 * random_multiplier2

                    if actual_status[n1] > 1:
                        #print("Error out of bounds for n1", actual_status[n1])
                        actual_status[n1] = avg

                    if actual_status[n2] > 1 :
                        #print("Error out of bounds for n2", actual_status[n2])
                        actual_status[n2] = avg

                if self.params['model']['noise'] == 0:
                    avg = (actual_status[n1] + actual_status[n2]) / 2.0
                    #add another parameter mu here for convergence
                    actual_status[n1] = avg
                    actual_status[n2] = avg
                # se la rete è completa aggiorno all'interno del ciclo
                # self.sts, così lo riprendo sempre aggiornato
                if len(self.node_data) == 0:
                    self.sts[n1] = avg
                    self.sts[n2] = avg

        # delta, node_count, status_delta = self.status_delta(actual_status)
        delta = actual_status
        node_count = {}
        status_delta = {}

        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta,
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

    def steady_state(self, max_iterations=100000, nsteady=1000, sensibility=0.00001, node_status=True,
                     progress_bar=False):
        """
        Execute a bunch of model iterations
        :param max_iterations: the maximum number of iterations to execute
        :param nsteady: number of required stable states
        :param sensibility: sensibility check for a steady state
        :param node_status: if the incremental node status has to be returned.
        :param progress_bar: whether to display a progress bar, default False
        :return: a list containing for each iteration a dictionary {"iteration": iteration_id, "status": dictionary_node_to_status}
        """
        system_status = []
        steady_it = 0
        for it in tqdm.tqdm(range(0, max_iterations), disable=not progress_bar):
            its = self.iteration(node_status)

            if it > 0:
                old = np.array(list(system_status[-1]['status'].values()))
                actual = np.array(list(its['status'].values()))
                res = np.abs(old - actual)
                if np.all((res < sensibility)):
                    steady_it += 1
                else:
                    steady_it = 0

            system_status.append(its)
            if steady_it == nsteady:
                return system_status[:-nsteady]

        return system_status