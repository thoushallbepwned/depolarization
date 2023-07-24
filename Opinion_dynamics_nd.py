#This code will take existing graph structures and seed them with opinions according to the specifications in the code.
#In particular, we will be seeding a graph with multidimensional opions and then running an opinion dynamics model on it.

# Import Standard Libraries
from collections import Counter
import os
import pickle
import warnings

# Import Third Party Libraries
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as opn
import networkx as nx
import numpy as np
import pandas as pd
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
from tqdm import tqdm

# Import Local Libraries
from Homophily_generated_networks import *
from Modified_algorithmic_bias_nd import *
from polarization_metric import *


warnings.filterwarnings("ignore")

# Network topology
# parameters governing the graph structure
"These variables should remain constant for all experimental runs"
n = 1000  # number of nodes Note: This should be an even number to ensure stability
m = 8  # number of edges per node
p = 0.70  # probability of rewiring each edge
minority_fraction = 0.5  # fraction of minority nodes in the network
similitude = 0.8  # similarity metric
d = 4  # number of dimension

#defining helper functions
def Extract(lst):
    return [item[0] for item in lst]


def generate_title(config):
    model_parameters = config.get_model_parameters()

    title = (
        f"mode: {model_parameters['operational_mode']}, "
        f"epsilon: {np.round(model_parameters['epsilon'], 2)}, "
        f"intervention: {model_parameters['link_prediction']}, "
        f"noise: {model_parameters['noise']}, "
        f"dm: {model_parameters['distance_method']}"
    )
    return title

def run_simulation(distance_method, mode, epsilon, operational_mode):

    # Generating graph
    g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

    # Model selection
    model = AlgorithmicBiasModel_nd(g)


    "These variables are editable for each experimental run"
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", epsilon) #bounded confidence parameter
    config.add_model_parameter("mu", 0.5) #convergence parameter
    config.add_model_parameter("gamma", 0) #bias parameter
    config.add_model_parameter("mode", mode) #initial opinion distribution
    config.add_model_parameter("noise", 0.05) # noise parameter that cannot exceed 10%
    config.add_model_parameter("minority_fraction", minority_fraction) # minority fraction in the network
    config.add_model_parameter("dims", d) # number of dimensions
    config.add_model_parameter("gamma_cov", 0.35) # correlation between dimensions
    config.add_model_parameter("distance_method", distance_method) # fraction of minority nodes in the network
    config.add_model_parameter("fixed", True) # distribution opinion parameter
    config.add_model_parameter("operational_mode", operational_mode) # operational parameter
    config.add_model_parameter("link_prediction", intervention) # link prediction parameter
    model.set_initial_status(config)


    # Simulation execution
    epochs = 16
    if operational_mode == "ensemble":
        epochs = int(epochs/d)
    else:
        epochs = epochs
    #Simulation execution

    iterations = model.iteration_bunch(epochs, node_status=True, progress_bar=False)

    print("What exactly is iterations?", len(iterations))


    test_vector = iterations[0]['status']
    control_graph = g.copy()
    #print("Initial distribution", test_vector)

    # assigning opinions to nodes
    for nodes in control_graph.nodes:
         control_graph.nodes[nodes]['opinion'] = test_vector[nodes]

    x_before =nx.get_node_attributes(control_graph, 'opinion')
    pickle.dump(control_graph, open(f"graphs/2k_nodes/before_graph_{operation}_{method}_{seed}_{np.round(epsilon,2)}.p", "wb"))
    array_int = list(x_before.values())
    data_1 =np.array(array_int)
    opinion_vector = iterations[epochs-1]['status']

    for nodes in g.nodes:
         g.nodes[nodes]['opinion'] = opinion_vector[nodes]

    # visualization


    "Visualization before and after "
    #print("what exactly is g?", g)
    #print("what is the type", type(g))
    pickle.dump(g, open(f"graphs/final_graph_{operation}_{method}_{seed}_{np.round(epsilon,2)}.p", "wb"))
    x_after =nx.get_node_attributes(g, 'opinion')
    array_int_after = list(x_after.values())
    data_2 =np.array(array_int_after)

    df_before = pd.DataFrame(data_1, columns=[f'Dimension {i + 1}' for i in range(d)])
    df_after = pd.DataFrame(data_2, columns=[f'Dimension {i + 1}' for i in range(d)])

    polarization_before = polarization_metric(pd.DataFrame(data_1, columns=[f'Dimension {i + 1}' for i in range(d)]))
    polarization_after = polarization_metric(pd.DataFrame(data_2, columns=[f'Dimension {i + 1}' for i in range(d)]))


    "testing what this fucking data_1 and data_2 are"

    print(data_1.shape, data_2.shape)

    # Plot histograms
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    mean_polarization_before = np.mean(polarization_before)
    mean_polarization_after = np.mean(polarization_after)

    # Calculate depolarization
    depolarization = 100 - (mean_polarization_after/mean_polarization_before)*100
    polarization_decrease = polarization_before - polarization_after
    net_depolarization = np.sum(polarization_decrease)


    # Calculate maximum decrease
    max_decrease = np.max(polarization_decrease)
    max_decrease_dim = np.argmax(polarization_decrease)+1


    # Plot histograms
    for i, col in enumerate(df_before.columns):
        axes[i].hist(df_before[col], bins=20, alpha=0.5, label='Before Opinion Dynamics', color='blue')
        axes[i].hist(df_after[col], bins=20, alpha=0.5, label='After Opinion Dynamics', color='green')
        axes[i].set_title(f'Histogram for {col}')
        axes[i].text(0.05, -0.2, f'P before: {polarization_before[i]:.2f}', transform=axes[i].transAxes, fontsize=10)
        axes[i].text(0.45, -0.2, f'P after: {polarization_after[i]:.2f}', transform=axes[i].transAxes, fontsize=10)
        axes[i].legend()
        axes[i].set_xlim(-1, 1)
    plt.suptitle(generate_title(config), fontsize=12)
    plt.figtext(0.2, 0.01, f'Depolarization percentage: {depolarization:.2f}', ha="center", fontsize=10,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.figtext(0.50, 0.01, f'Net depolarization: {net_depolarization:.2f}', ha="center",
                fontsize=10, bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})
    plt.figtext(0.80, 0.01, f'Max Decrease: {max_decrease:.2f} in Dimension: {max_decrease_dim}', ha="center",
                fontsize=10, bbox={"facecolor": "lightgreen", "alpha": 0.5, "pad": 5})

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)


    a = plt.gcf()

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
        #axes[d_index].legend()
        axes[d_index].set_ylim(-1, 1)


    plt.suptitle(generate_title(config), fontsize=12)
    plt.xlabel('Iterations')
    fig.tight_layout()
    b = plt.gcf()

    return a, b


if __name__ == "__main__":
    interval = np.arange(0, 1.1, 0.2)
    dims = 4

    noise = ["noiseless"]
    operation_list = ["sequential"]#, "soft_max", "ensemble"] #["ensemble"]#["ensemble","bounded"] #["softmax", "sequential",
    method_list = ["strict_euclidean"]#, "mean_euclidean"]#, "strict_euclidean", "cosine", "size_cosine"]
    seeding_list = ["mixed"]#, "normal", "polarized"]
    intervention_status = ["intervened", "natural"]


    for noise_mode in noise:
        for operation in tqdm(operation_list):
            print(f"\nCurrently operating {operation} simulations\n")
            os.makedirs(f"images/{noise_mode}/{operation}", exist_ok=True)
            for method in tqdm(method_list):
                print(f"For {operation} mode running the {method} method\n")
                os.makedirs(f"images/{noise_mode}/{operation}/{method}", exist_ok=True)
                for seed in seeding_list:
                    print(f"seeding mode is {seed} for {operation} and {method}\n")
                    os.makedirs(f"images/{noise_mode}/{operation}/{method}/{seed}", exist_ok=True)
                    for intervention in intervention_status:
                        print(f"The current intervention status is {intervention}\n")
                        os.makedirs(f"images/{noise_mode}/{operation}/{method}/{seed}/{intervention}", exist_ok=True)
                        if __name__ == "__main__":
                            # same code as above...
                            for i in interval:
                                fig1, fig2 = run_simulation(method, seed, i, operation)
                                index = np.round(i, 2)

                                fig1.savefig(
                                    f"images/{noise_mode}/{operation}/{method}/{seed}/{intervention}/fig1_{index}.png",
                                    dpi=fig1.dpi)
                                fig2.savefig(
                                    f"images/{noise_mode}/{operation}/{method}/{seed}/{intervention}/fig2_{index}.png",
                                    dpi=fig2.dpi)
