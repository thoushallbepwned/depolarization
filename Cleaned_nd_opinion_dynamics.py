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
n = 2000  # number of nodes Note: This should be an even number to ensure stability
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

def configure_model(epsilon, mode, minority_fraction, d, operational_mode, distance_method, intervention):


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

    return config

def calculate_epochs(operational_mode, d):
    if operational_mode == "ensemble":
        epochs = int(8/d)
    else:
        epochs = 8
    return epochs

def get_control_graph(iterations, g):
    test_vector = iterations[0]['status']
    control_graph = g.copy()
    #print("Initial distribution", test_vector)

    # assigning opinions to nodes
    for nodes in control_graph.nodes:
         control_graph.nodes[nodes]['opinion'] = test_vector[nodes]
    return control_graph

def get_final_graph(iterations, g, epochs):
    opinion_vector = iterations[epochs-1]['status']

    for nodes in g.nodes:
         g.nodes[nodes]['opinion'] = opinion_vector[nodes]
    return g

def get_data_frames(control_graph, g, d):
    # Extract the opinions before simulation
    x_before = nx.get_node_attributes(control_graph, 'opinion')
    array_int = list(x_before.values())
    data_1 = np.array(array_int)

    # Extract the opinions after simulation
    x_after = nx.get_node_attributes(g, 'opinion')
    array_int_after = list(x_after.values())
    data_2 = np.array(array_int_after)

    # Create the DataFrames
    df_before = pd.DataFrame(data_1, columns=[f'Dimension {i + 1}' for i in range(d)])
    df_after = pd.DataFrame(data_2, columns=[f'Dimension {i + 1}' for i in range(d)])

    return df_before, df_after

def get_polarization_metrics(df_before, df_after):
    # Calculate polarization metrics
    polarization_before = polarization_metric(df_before)
    polarization_after = polarization_metric(df_after)
    return polarization_before, polarization_after


def visualize_line_plot(metrics_results):
    metric_names = ['mean_polarization_before', 'mean_polarization_after', 'depolarization', 'net_depolarization']

    #print("this is metric_results", metrics_results)
    plt.figure()

    fig, axs = plt.subplots(2, figsize=(10, 6 * 2))  # Set the number of plots to 2 because we have 2 metrics to plot

    metrics_data = {
        'depolarization': [],
        'net_depolarization': []
    }

    for intervention, results in metrics_results.items():
        epsilon_values = []  # Create a list to hold all epsilon values
        depolarizations = []
        net_depolarizations = []

        for i, result in enumerate(results):
            epsilon, metrics_array = result
            epsilon_values.append(epsilon)  # Store this epsilon value

            if i == 0:  # initial state, 'polarization_before'
                polarization_before = metrics_array
                mean_polarization_before = np.mean(polarization_before)
            else:  # 'polarization_after'
                polarization_after = metrics_array
                mean_polarization_after = np.mean(polarization_after)

                # calculate 'depolarization' and 'net_depolarization'
                depolarization = 100 - (mean_polarization_after / mean_polarization_before) * 100
                polarization_decrease = polarization_before - polarization_after
                net_depolarization = np.sum(polarization_decrease)

                depolarizations.append(depolarization)
                net_depolarizations.append(net_depolarization)

        # Store the data for later plotting
        metrics_data['depolarization'].append((epsilon_values[1:], depolarizations, intervention))
        metrics_data['net_depolarization'].append((epsilon_values[1:], net_depolarizations, intervention))

    # Now plot 'depolarization' and 'net_depolarization' for each intervention method on separate plots
    for ax, (metric_name, data) in zip(axs, metrics_data.items()):
        for epsilon_values, metric_values, intervention in data:
            ax.plot(epsilon_values, metric_values, label=intervention)
        ax.set_xlabel('Epsilon')
        ax.set_ylabel(metric_name)
        ax.legend()

    plt.show()
    plt.close()
    return fig

def visualize_histogram(results):

    plt.figure()
    num_interventions = len(results)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8 * num_interventions))
    axes = axes.ravel()


    df_before = results["natural"]["df_before"]
    df_after_natural = results["natural"]["df_after"]
    df_after_intervened = results["predicted"]["df_after"]
    df_after_targeted = results["removal"]["df_after"]  # Added targeted
    polarization_before = results["natural"]["polarization_before"]
    polarization_after = results["natural"]["polarization_after"]
    config = results["natural"]["config"]

    mean_polarization_before = np.mean(polarization_before)
    mean_polarization_after = np.mean(polarization_after)

    depolarization = 100 - (mean_polarization_after / mean_polarization_before) * 100
    polarization_decrease = polarization_before - polarization_after
    net_depolarization = np.sum(polarization_decrease)

    max_decrease = np.max(polarization_decrease)
    max_decrease_dim = np.argmax(polarization_decrease) + 1

    for j, col in enumerate(df_before.columns):
        axes[j].hist(df_before[col], bins=20, alpha=0.5, label='Before Opinion Dynamics', color='blue')
        axes[j].hist(df_after_natural[col], bins=20, alpha=0.5, label='After Opinion Dynamics (Natural)', color='green')
        axes[j].hist(df_after_intervened[col], bins=20, alpha=0.5, label='After Opinion Dynamics (Predicted)',
                     color='red')
        axes[j].hist(df_after_targeted[col], bins=20, alpha=0.5, label='After Opinion Dynamics (Removal)',
                     color='yellow')  # Added targeted
        axes[j].set_title(f'Histogram for {col} with')
        axes[j].text(0.05, -0.2, f'P before: {polarization_before[j]:.2f}', transform=axes[j].transAxes, fontsize=10)
        axes[j].text(0.45, -0.2, f'P after: {polarization_after[j]:.2f}', transform=axes[j].transAxes, fontsize=10)
        axes[j].legend()
        axes[j].set_xlim(-1, 1)

    plt.suptitle(generate_title(config), fontsize=12)
    plt.figtext(0.2, 0.01, f'Depolarization percentage: {depolarization:.2f}', ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.figtext(0.50, 0.01, f'Net depolarization: {net_depolarization:.2f}', ha="center", fontsize=10, bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})
    plt.figtext(0.80, 0.01, f'Max Decrease: {max_decrease:.2f} in Dimension: {max_decrease_dim}', ha="center", fontsize=10, bbox={"facecolor": "lightgreen", "alpha": 0.5, "pad": 5})

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()
    return fig


"Running the main simulation function"
def run_simulation(distance_method, mode, epsilon, operational_mode, intervention_status):

    results = {}



    # Initialize lists to hold metrics for each simulation run
    mean_polarization_before_results = {intervention: [] for intervention in intervention_status}
    mean_polarization_after_results = {intervention: [] for intervention in intervention_status}
    depolarization_results = {intervention: [] for intervention in intervention_status}
    net_depolarization_results = {intervention: [] for intervention in intervention_status}
    max_decrease_results = {intervention: [] for intervention in intervention_status}

    for intervention in intervention_status:

        print("running in intervention mode: ", intervention)


        # Generating graph
        g = homophilic_barabasi_albert_graph(n, m, minority_fraction, similitude, p) # generating Graph

        # Model selection
        model = AlgorithmicBiasModel_nd(g)

        # Configuring model
        config = configure_model(epsilon, mode, minority_fraction, d, operational_mode, distance_method, intervention)

        # Setting initial status
        model.set_initial_status(config)

        # Simulation execution
        epochs = calculate_epochs(operational_mode, d)
        iterations = model.iteration_bunch(epochs, node_status=True, progress_bar=False)

        # Control graph
        control_graph = get_control_graph(iterations, g)

        # Graph after iterations
        g = get_final_graph(iterations, g, epochs)

        # if intervention == "natural":
        #
        #     #pickle.dump(control_graph, open(f"natural/{n}_nodes/before_graph_{operation}_{method}_{seed}_{np.round(epsilon,2)}_{noise_mode}.p", "wb"))
        #     pickle.dump(g, open(f"natural/{n}_nodes/final_graph_{operation}_{method}_{seed}_{np.round(epsilon,2)}_{noise_mode}.p", "wb"))

        # # DataFrames for visualization
        df_before, df_after = get_data_frames(control_graph, g, d)

        # Polarization metrics
        polarization_before, polarization_after = get_polarization_metrics(df_before, df_after)

        # Store the results for later use
        metric_results[intervention].append((epsilon, polarization_after))


        # Instead of creating the figure, store the results
        results[intervention] = {
            "df_before": df_before,
            "df_after": df_after,
            "polarization_before": polarization_before,
            "polarization_after": polarization_after,
            "config": config
        }

        mean_polarization_before = np.mean(polarization_before)
        mean_polarization_after = np.mean(polarization_after)
        depolarization = 100 - (mean_polarization_after / mean_polarization_before) * 100
        polarization_decrease = polarization_before - polarization_after
        net_depolarization = np.sum(polarization_decrease)
        max_decrease = np.max(polarization_decrease)

        # Store the metrics for this run
        mean_polarization_before_results[intervention].append((epsilon, mean_polarization_before))
        mean_polarization_after_results[intervention].append((epsilon, mean_polarization_after))
        depolarization_results[intervention].append((epsilon, depolarization))
        net_depolarization_results[intervention].append((epsilon, net_depolarization))
        max_decrease_results[intervention].append((epsilon, max_decrease))

        metrics_results = {
            'mean_polarization_before': mean_polarization_before_results,
            'mean_polarization_after': mean_polarization_after_results,
            'depolarization': depolarization_results,
            'net_depolarization': net_depolarization_results,
            'max_decrease': max_decrease_results,
        }

    return results, metric_results




if __name__ == "__main__":
    interval = np.arange(0.25, 0.80, 0.05)
    dims = 4

    noise = ["noisy"]#,"noiseless"]
    operation_list = ["sequential", "softmax","bounded"]
    method_list = ["mean_euclidean", "strict_euclidean", "cosine", "size_cosine"]
    seeding_list = ["mixed"]#, "normal", "polarized"]
    intervention_status = ["natural", "predicted", "removal"]#["natural", "intervened", "targeted"]


    for noise_mode in noise:
        for operation in tqdm.tqdm(operation_list):
            print(f"\nCurrently operating {operation} simulations\n")
            os.makedirs(f"images/{noise_mode}/{operation}", exist_ok=True)
            for method in tqdm.tqdm(method_list):
                print(f"For {operation} mode running the {method} method\n")
                os.makedirs(f"images/{noise_mode}/{operation}/{method}", exist_ok=True)
                for seed in seeding_list:
                    print(f"seeding mode is {seed} for {operation} and {method}\n")
                    os.makedirs(f"images/{noise_mode}/{operation}/{method}/{seed}", exist_ok=True)

                    if __name__ == "__main__":

                        metric_results = {"natural": [], "predicted": [], "removal": []}
                        for i in interval:

                            i = np.round(i, 2)

                            results, metric_results = run_simulation(method, seed, i, operation, intervention_status)
                            fig = visualize_histogram(results)
                            index = np.round(i, 2)

                            fig.savefig(
                                f"images/{noise_mode}/{operation}/{method}/{seed}/fig1_{index}.png",
                                dpi=fig.dpi)

                        fig2 = visualize_line_plot(metric_results)
                        fig2.savefig(
                            f"images/{noise_mode}/{operation}/{method}/{seed}/fig2_{index}.png",
                            dpi=fig.dpi)
