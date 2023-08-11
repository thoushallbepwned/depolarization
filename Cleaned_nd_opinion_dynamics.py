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


import matplotlib.pyplot as plt


def visualize_natural_state_line_plot(data, operation_list, title):
    """
    Visualizes the polarization metrics for the natural state's first method across different operations in a 1x3 plot.

    Args:
    - data (dict): The dictionary containing the polarization metrics.
    - operation_list (list): List of operations to visualize.

    Returns:
    - A line graph visualization.
    """
    print("visualizing now")
    def compute_advanced_metrics(metrics_results):
        advanced_metrics = {}
        for intervention, results in metrics_results.items():
            epsilon_values = []
            depolarizations = []
            net_depolarizations = []
            max_contributions = []

            for i, result in enumerate(results[0]):
                epsilon, metrics_array = result
                epsilon_values.append(epsilon)

                if i == 0:  # initial state
                    polarization_before = metrics_array
                    mean_polarization_before = np.sum(polarization_before)
                else:  # 'polarization_after'
                    polarization_after = metrics_array
                    mean_polarization_after = np.sum(polarization_after)

                    depolarization = 100 - (mean_polarization_after / mean_polarization_before) * 100
                    polarization_decrease = polarization_before - polarization_after
                    net_depolarization = np.mean(polarization_decrease)

                    max_contribution = max(polarization_after) / np.sum(polarization_after) * 100
                    max_contributions.append(max_contribution)

                    depolarizations.append(depolarization)
                    net_depolarizations.append(net_depolarization)

            advanced_metrics[intervention] = {
                "epsilon_values": epsilon_values[1:],
                "depolarizations": depolarizations,
                "net_depolarizations": net_depolarizations,
                "max_contributions": max_contributions
            }

        return advanced_metrics

    advanced_metrics_results = compute_advanced_metrics(data)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 grid due to the new metric

    fig.suptitle(f"Metrics collected for {title} mode", fontsize=20)
    for operation in operation_list:
        x_axis = [entry[0] for entry in data[operation][0]]
        natural_polarizations = [sum(entry[1]) for entry in data[operation][0]]  # Sum of all elements in the array

        # Plotting natural polarizations
        axes[0, 0].plot(x_axis, natural_polarizations, marker='o', label=f'{operation}')

        # Plotting depolarizations
        axes[0, 1].plot(advanced_metrics_results[operation]["epsilon_values"],
                        advanced_metrics_results[operation]["depolarizations"], marker='x',
                        label=f'{operation}')

        # Plotting net depolarizations
        axes[1, 0].plot(advanced_metrics_results[operation]["epsilon_values"],
                        advanced_metrics_results[operation]["net_depolarizations"], marker='s',
                        label=f'{operation}')

        # New subplot for max contribution
        axes[1, 1].plot(advanced_metrics_results[operation]["epsilon_values"],
                        advanced_metrics_results[operation]["max_contributions"], marker='d',
                        label=f'{operation}')


    # Setting titles, labels, and other aesthetics
    titles = [f'{title} State Polarization', '% Depolarization', 'Mean Depolarization', 'Max Contribution %']
    for i, ax in enumerate(axes.ravel()):  # .ravel() flattens the 2x2 array to a 1D array for iteration
        ax.set_title(titles[i], fontsize=16)
        ax.set_xlabel('Epsilon value', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    plt.show()

    return fig


def run_simulation(distance_method, mode, epsilon, operational_mode, intervention_status):

    results = {}



    # Initialize lists to hold metrics for each simulation run
    mean_polarization_before_results = {intervention: [] for intervention in intervention_status}
    mean_polarization_after_results = {intervention: [] for intervention in intervention_status}
    depolarization_results = {intervention: [] for intervention in intervention_status}
    net_depolarization_results = {intervention: [] for intervention in intervention_status}
    max_decrease_results = {intervention: [] for intervention in intervention_status}

    for intervention in intervention_status:

        #print("running in intervention mode: ", intervention)


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
    interval = np.arange(0.10, 0.90, 0.05)
    dims = 4

    noise = ["noisy"]#,"noiseless"]
    operation_list = ["sequential", "softmax","bounded", "ensemble"]
    method_list = ["cosine"]#, "size_cosine"]#["mean_euclidean"]#, "strict_euclidean", "cosine", "size_cosine"]
    seeding_list = ["mixed"]#, "normal", "polarized"]
    intervention_status = ["natural", "predicted", "high-removal", "low-removal"]#["natural", "intervened", "targeted"]

    operation_results_nat = {"sequential": [], "softmax": [], "bounded": [], "ensemble": []}
    operation_results_pred = {"sequential": [], "softmax": [], "bounded": [], "ensemble": []}
    operation_results_Hrem = {"sequential": [], "softmax": [], "bounded": [], "ensemble": []}
    operation_results_Lrem = {"sequential": [], "softmax": [], "bounded": [], "ensemble": []}

    def subtract_arrays(array1, array2):
        return array1 - array2

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

                    metric_results = {"natural": [], "predicted": [], "high-removal": [], "low-removal": []}

                    for i in interval:
                        print("Currently running epsilon: ", i)
                        i = np.round(i, 2)

                        results, metric_results = run_simulation(method, seed, i, operation, intervention_status)
                    #     fig = visualize_histogram(results)
                    #     index = np.round(i, 2)
                    #
                    #     fig.savefig(
                    #         f"images/{noise_mode}/{operation}/{method}/{seed}/fig1_{index}.png",
                    #         dpi=fig.dpi)
                    #
                    # fig2 = visualize_line_plot(metric_results)
                    # fig2.savefig(
                    #     f"images/{noise_mode}/{operation}/{method}/{seed}/fig2_{index}.png",
                    #     dpi=fig.dpi)

                    operation_results_nat[operation].append(metric_results["natural"])
                    operation_results_pred[operation].append(metric_results["predicted"])
                    operation_results_Hrem[operation].append(metric_results["high-removal"])
                    operation_results_Lrem[operation].append(metric_results["low-removal"])

                    print(operation_results_nat)
                    print(operation_results_Lrem)

        # Compute the delta values
        delta_Hrem = {}
        delta_Lrem = {}

        for operation in operation_list:
            nat_data = operation_results_nat[operation][0]
            hrem_data = operation_results_Hrem[operation][0]
            lrem_data = operation_results_Lrem[operation][0]

            # Note the order of subtraction: hrem_arr - nat_arr and lrem_arr - nat_arr
            delta_Hrem_data = [(eps, subtract_arrays(hrem_arr, nat_arr)) for (eps, nat_arr), (_, hrem_arr) in
                               zip(nat_data, hrem_data)]
            delta_Lrem_data = [(eps, subtract_arrays(lrem_arr, nat_arr)) for (eps, nat_arr), (_, lrem_arr) in
                               zip(nat_data, lrem_data)]

            delta_Hrem[operation] = [delta_Hrem_data]
            delta_Lrem[operation] = [delta_Lrem_data]

        # Visualize the delta values
        fig7 = visualize_natural_state_line_plot(delta_Hrem, operation_list, "Delta High Removal")
        fig8 = visualize_natural_state_line_plot(delta_Lrem, operation_list, "Delta Low Removal")

                #print("outer loop", operation_results)
        fig3 = visualize_natural_state_line_plot(operation_results_nat, operation_list, "Natural")
        fig4 = visualize_natural_state_line_plot(operation_results_Hrem, operation_list, "High Removal")
        fig5 = visualize_natural_state_line_plot(operation_results_pred, operation_list, "Predicted")
        fig6 = visualize_natural_state_line_plot(operation_results_Lrem, operation_list, "Low Removal")

        fig3.savefig(
            f"polarization logs/{method}_natural_state.png",
            dpi=fig3.dpi)
        fig4.savefig(
            f"polarization logs/{method}_high_removal.png",
            dpi=fig4.dpi)
        fig5.savefig(
            f"polarization logs/{method}_predicted_state.png",
            dpi=fig5.dpi)
        fig6.savefig(
            f"polarization logs/{method}_low_removal.png",
            dpi=fig6.dpi)

        fig7.savefig(
            f"polarization logs/{method}_delta_high_removal.png",
            dpi=fig7.dpi)
        fig8.savefig(
            f"polarization logs/{method}_delta_low_removal.png",
            dpi=fig8.dpi)

