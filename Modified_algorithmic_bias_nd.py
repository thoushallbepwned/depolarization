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
import mpl_toolkits.mplot3d as Axes3D


__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = ["alina.sirbu@unipi.it", "giulio.rossetti@isti.cnr.it", "valentina.pansanella@sns.it"]
#modified by Paul Verhagen


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

        def Extract(lst):
            return [item[0] for item in lst]

        def Match(a, b):
            return [elem for elem in a if elem in b]

        def filter(array1, z):
            return [tuple(elem for i, elem in enumerate(tup) if i in z) for tup in array1]

        def polarized_distr_nd(G, n, minority_fraction, d, gamma):
            lower, upper = -1, 1  # lower and upper bounds

            count2 = int(n * minority_fraction)
            count1 = int(n * (1 - minority_fraction))

            s = np.zeros((n, d))
            correlation_matrix = np.identity(d) * (1 - gamma) + gamma
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
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 3], cmap=plt.hot())
            fig.colorbar(img)
            plt.show()


            return s

        def normal_distr_nd(G, n, d, gamma):

            lower, upper = -1, 1  # lower and upper bounds
            s = np.zeros((n, d))

            correlation_matrix = np.identity(d) * (1 - gamma) + gamma
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
                    print(cov)
                    s = np.random.multivariate_normal(mu, cov, n)
                    s = s / np.max(s)

            if d == 1:
                mu = np.random.uniform(low=-0.25, high=0.25, size=d)
                sigma = np.random.uniform(low=0.1, high=0.25, size=d)  # mean and standard deviation
                X = stats.truncnorm(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

                truncacted_array = X.rvs(n)
                s = np.reshape(truncacted_array, (n, 1))
            return (s)

        def mixed_distr_nd(G, n, minority_fraction, d, gamma):
            print("Applying the mixed sampling regime, the following choices were made:")
            lower, upper = -1, 1  # lower and upper bounds

            count2 = int(n * minority_fraction)
            count1 = int(n * (1 - minority_fraction))


            s = np.zeros((n, d))
            correlation_matrix = np.identity(d) * (1 - gamma) + gamma
            # print(correlation_matrix)
            options = ["a", "b"]
            for i in range(d):
                choices = random.choices(options, weights=[1, 1], k=1)
                # print("what did we pick?", choices)

                if choices[0] == "a":
                    print("going polarized")
                    s_1 = polarized_distr_nd(G, n, minority_fraction, 1, gamma)
                    plt.hist(s_1, range=(-1, 1), bins=50)
                    plt.show()
                else:
                    print("going normal")
                    s_1 = normal_distr_nd(G, n, 1, gamma)
                    plt.hist(s_1, range=(-1, 1), bins=50)
                    plt.show()
                s[:, i] = s_1[:, 0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 3], cmap=plt.hot())
            fig.colorbar(img)
            plt.show()

            return s

        # set node status

        #setting node sequence based on attributes
        array = nx.get_node_attributes(self.graph, 'color')
        sorted_array = sorted(array.items(), key=lambda x: x[1])
        index_list = Extract(sorted_array)

        if self.params['model']['noise'] == 0:
            print("running in noiseless mode")
        else:
            print("running in noisy mode")

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
            s = normal_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['dims'], self.params['model']['gamma'])
            #print("this is the shape of s", s, s.shape)

            sorted_dist = np.sort(s, axis = 0)
            #print(sorted_dist, sorted_dist.shape)
            #print( np.mean(s, axis = 0), np.mean((s+2), axis = 0))

            i = 0
            for node in index_list:
                entry = sorted_dist[i].flatten()
                if self.params['model']['dims'] == 1:

                    self.status[node] = float(entry)
                else:
                    self.status[node] = entry.tolist()
                #print(type(entry))
                i += 1
            #print("Wtf is this?", self.status)
            #print("the shape is", self.status.items())
            self.initial_status = self.status.copy()

        if self.params['model']['mode'] == 'polarized':
            print("set to polarized mode")

            dist = polarized_distr_nd(self.graph, len(self.graph.nodes()), self.params['model']['minority_fraction'],self.params['model']['dims'], self.params['model']['gamma'])
            #dist = [(2*x) -1 for x in dist]
            sorted_dist = sorted(dist)


            sorted_dist_round = np.round(sorted_dist)
            plt.hist(dist, range = (-1,1), bins = 50)
            plt.show( )

            i = 0
            for node in index_list:

                self.status[node] = sorted_dist[i]
                #print(self.status[node])

                i += 1

            self.initial_status = self.status.copy()



        ### Initialization numpy representation

        max_edgees = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2
        #print("what is this structure", self.status.items())
        #print("what is nids?", np.array(list(self.status.items())))

        if self.params['model']['dims'] == 1:
            nids = np.array(list(self.status.items()))
            self.ids = nids[:, 0]
        else:
            nids = list(self.status.items())
            self.ids = Extract(nids)

        #print("what are nids exactly?", nids)

        #print("and what do we need for the ids?", self.ids)
        # self.ids = np.array(len(self.graph.nodes()))

        if max_edgees == self.graph.number_of_edges():
            self.sts = nids[:, 1]
            #print("what is sts?", self.sts)

        else:
            "Found the location where a bunch of things need to change"

            for i in self.graph.nodes:
                if self.params['model']['dims'] == 1:
                    i_neigh = list(self.graph.neighbors(i))
                    i_ids = nids[:, 0][i_neigh]
                    i_sts = nids[:, 1][i_neigh]
                    #print("what are i_neigh?", i_neigh)
                    #print("what are i_ids?", i_ids)
                    #print(np.array_equal(i_ids, i_neigh))
                    #print("what are i_ists?", i_sts)

                    self.node_data[i] = (i_ids, i_sts)
                else:
                    i_neigh = list(self.graph.neighbors(i))
                    i_ids = i_neigh
                    #i_ids = [tup[0] for tup in nids if tup[0] in i_neigh]
                    i_sts = [tup[1] for tup in nids if tup[0] in i_neigh]
                    #print("what are i_neigh?", i_neigh)
                    #print("what are i_ids?", i_ids)
                    #print(np.array_equal(i_ids, i_neigh))
                    #print("what are i_ists?", i_sts)

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

        #testing the assortivity:
        #print("getting node attrtibudes", nx.get_node_attributes(self.graph, 'status'))
        #print("assortivity before opinion dynamics", nx.numeric_assortativity_coefficient(self.graph, 'status'))

        "Adding a very large forloop here that will iterate over dimensions"


        #interact with peers
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
            #if selection_prob == 0:

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



            "Stuff gets interesting here, because we will need to specify a distance metric here to govern the interaction"

            if self.params['model']['dims'] > 1:
                diff = [abs((actual_status[n1][i]+2) - (actual_status[n2][i]+2)) for i in range(self.params['model']['dims'])]
            else:
                diff = np.abs((actual_status[n1]+2) - (actual_status[n2]+2))

            diff = np.array(diff)


            # update status of n1 and n2
            #diff = np.abs((np.array(actual_status[n1]+2)) - np.array((actual_status[n2]+2)))
            # Testing whether epsilon is respected
            # if diff > self.params['model']['epsilon']:
            #     print("ERROR: Node selection opinion out of bounds")

            "First model: We are going to iteratate over all admissible dimension"
            if self.params['model']['dims'] > 1:
                for i in range(self.params['model']['dims']):

                    if float(diff[i]) < self.params['model']['epsilon']:
                        # Adding a little bit of extra noise into the equation
                        if self.params['model']['noise'] > 0:
                            change1 = ((actual_status[n2][i]+2) - (actual_status[n1][i]+2))
                            change2 = ((actual_status[n1][i]+2) - (actual_status[n2][i]+2))


                            noise1 = np.random.uniform(low=-1*change1*self.params['model']['noise'], high=change1*self.params['model']['noise'])
                            noise2 = np.random.uniform(low=-1*change2*self.params['model']['noise'], high=change2*self.params['model']['noise'])


                            actual_status[n1][i] = actual_status[n1][i]+ self.params['model']['mu']*change1 + noise1
                            actual_status[n2][i] = actual_status[n2][i]+ self.params['model']['mu']*change2 + noise2




                        if self.params['model']['noise'] == 0:

                            # Testing some ways to see if the absolute difference is screwing things up
                            change1 = (actual_status[n2][i]+10) - (actual_status[n1][i]+10)
                            change2 = (actual_status[n1][i]+10) - (actual_status[n2][i]+10)
                            pos1 = actual_status[n1][i]
                            pos2 = actual_status[n2][i]

                #########################################
                            #THIS EQUATION IS SUPER IMPORTANT

                            #if actual_status[n1] > actual_status[n2]:
                            actual_status[n1][i] = pos1 + self.params['model']['mu']*change1
                            actual_status[n2][i] = pos2 + self.params['model']['mu']*change2


                        if len(self.node_data) == 0:
                            self.sts[n1] = actual_status[n1]
                            self.sts[n2] = actual_status[n2]
            else:
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
        delta, node_count, status_delta = self.status_delta(actual_status)
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