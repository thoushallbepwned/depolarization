import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def load_all_results(pickle_files):
    all_results = []
    for file in pickle_files:
        with open(file, 'rb') as f:
            all_results.append(pickle.load(f))
    return all_results

def aggregate_results(all_results, key):
    aggregated_data = []
    for results in all_results:
        if key in results:
            aggregated_data.append(results[key])
    return aggregated_data

def plot_net_difference(allowance):
    mpl.rcParams.update(mpl.rcParamsDefault)

    with open('simulation_results_42.pkl', 'rb') as f:
        loaded_results = pickle.load(f)

    # Provided options
    interactions = ["sequential", "softmax", "bounded", "ensemble"]
    scenarios = ['low-removal', 'high-removal']

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Determine the global y-axis limits for sum and % contribution
    all_y_sum_diff_values = []
    all_y_percent_diff_values = []

    for interaction in interactions:

        key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

        if key_to_retrieve not in loaded_results:
            continue

        value = loaded_results[key_to_retrieve]
        alt_dict_value = value[1]

        natural_values = alt_dict_value['natural']

        for scenario in scenarios:
            scenario_values = alt_dict_value[scenario]

            # For sum differences
            sum_diffs = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                         zip(scenario_values, natural_values)]
            all_y_sum_diff_values.extend(sum_diffs)

            # For % contribution differences
            percent_diffs = [
                max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                entry, natural_entry in zip(scenario_values, natural_values)]
            all_y_percent_diff_values.extend(percent_diffs)

    y_sum_diff_min = min(all_y_sum_diff_values) - 0.1
    y_sum_diff_max = max(all_y_sum_diff_values) + 0.1

    y_percent_diff_min = min(all_y_percent_diff_values) - 5
    y_percent_diff_max = max(all_y_percent_diff_values) + 5

    # Loop over each interaction option
    for interaction in interactions:
        #for allowance in ["strict_euclidean", "cosine"]:
        key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

        if key_to_retrieve not in loaded_results:
            continue

        value = loaded_results[key_to_retrieve]
        alt_dict_value = value[1]

        natural_values = alt_dict_value['natural']

        # Plot data for each scenario
        for i, scenario in enumerate(scenarios):
            scenario_values = alt_dict_value[scenario]

            # For sum differences
            x = [entry[0] for entry in scenario_values]
            y_diff = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                      zip(scenario_values, natural_values)]
            axs[0, i].plot(x, y_diff, '-o', label=f"{interaction} - {allowance}")
            axs[0, i].set_title(f"Sum Difference - {scenario}")
            axs[0, i].set_xlabel('Parameter Value')
            axs[0, i].set_ylabel('Net Sum Difference')
            axs[0, i].set_ylim(y_sum_diff_min, y_sum_diff_max)
            axs[0, i].legend()

            # For % contribution differences
            y_percent_diff = [
                max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                entry, natural_entry in zip(scenario_values, natural_values)]
            axs[1, i].plot(x, y_percent_diff, '-o', label=f"{interaction} - {allowance}")
            axs[1, i].set_title(f"% Contribution Difference - {scenario}")
            axs[1, i].set_xlabel('Parameter Value')
            axs[1, i].set_ylabel('Net % Contribution Difference')
            axs[1, i].set_ylim(y_percent_diff_min, y_percent_diff_max)
            axs[1, i].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.suptitle(f"Allowance: {allowance}", fontsize=16, y=1)
    #plt.show()
    #plt.show()
    return fig

def plot_allowance(allowance, all_results):
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Initialize empty list to store all loaded results
    #all_results = []

    # Load results from multiple pickle files
    # for filename in filenames:
    #     with open(filename, 'rb') as f:
    #         all_results.append(pickle.load(f))

    # Provided options
    interactions = ["sequential", "softmax", "bounded", "ensemble"]
    scenarios = ['natural', 'low-removal', 'high-removal']

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Determine the global y-axis limits for sum
    all_y_values = []

    # Determine the global y-axis limits for % contribution
    all_y_percent_values = []

    for interaction in interactions:
        all_sums = {scenario: [] for scenario in scenarios}
        all_percents = {scenario: [] for scenario in scenarios}

        for loaded_results in all_results:
            for param_value in [0.2]:
                key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', param_value)

                if key_to_retrieve not in loaded_results:
                    continue

                value = loaded_results[key_to_retrieve]
                alt_dict_value = value[1]

                for scenario in scenarios:
                    sums = [np.sum(entry[1]) for entry in alt_dict_value[scenario]]
                    percents = [max(entry[1]) / np.sum(entry[1]) * 100 for entry in alt_dict_value[scenario]]

                    all_sums[scenario].append(sums)
                    all_percents[scenario].append(percents)

        for scenario in scenarios:
            avg_sums = np.mean(all_sums[scenario], axis=0)
            var_sums = np.var(all_sums[scenario], axis=0)

            avg_percents = np.mean(all_percents[scenario], axis=0)
            var_percents = np.var(all_percents[scenario], axis=0)

            all_y_values.append(avg_sums)
            all_y_percent_values.append(avg_percents)

            # Plot data for each scenario
            x = [entry[0] for entry in alt_dict_value[scenario]]

            axs[0, scenarios.index(scenario)].errorbar(x, avg_sums, yerr=np.sqrt(var_sums), fmt='-o', label=interaction)
            axs[0, scenarios.index(scenario)].set_title(f"Sum - {scenario}")
            axs[0, scenarios.index(scenario)].set_xlabel('Parameter Value')
            axs[0, scenarios.index(scenario)].set_ylabel('Avg Sum of Metrics')

            axs[1, scenarios.index(scenario)].errorbar(x, avg_percents, yerr=np.sqrt(var_percents), fmt='-o', label=interaction)
            axs[1, scenarios.index(scenario)].set_title(f"% Contribution - {scenario}")
            axs[1, scenarios.index(scenario)].set_xlabel('Parameter Value')
            axs[1, scenarios.index(scenario)].set_ylabel('Avg % Contribution of Largest Value')

    for i in range(3):
        axs[0, i].legend()
        axs[1, i].legend()

    flattened_y_values = np.array(all_y_values).flatten()
    y_min = min(flattened_y_values) - 0.1
    y_max = max(flattened_y_values) + 0.1
    flat_y_perc = np.array(all_y_percent_values).flatten()
    y_percent_min = min(flat_y_perc) - 2
    y_percent_max = max(flat_y_perc) + 2

    for i in range(3):
        axs[0, i].set_ylim(y_min, y_max)
        axs[1, i].set_ylim(y_percent_min, y_percent_max)

    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.suptitle(f"Allowance: {allowance}", fontsize=16, y=1)
    plt.show()
    return fig


# def plot_allowance(allowance, pickle_files):
#     mpl.rcParams.update(mpl.rcParamsDefault)
#
#     with open('simulation_results_42.pkl', 'rb') as f:
#         loaded_results = pickle.load(f)
#     #print(loaded_results.keys())
#     # Provided options
#     interactions = ["sequential", "softmax", "bounded", "ensemble"]
#     scenarios = ['natural', 'low-removal', 'high-removal']
#
#     # Determine the global y-axis limits for sum
#     all_y_values = []
#
#     # Determine the global y-axis limits for % contribution
#     all_y_percent_values = []
#
#     # Extract all epsilon parameters
#     param_values = set([key[4] for key in loaded_results.keys() if key[2] == allowance])
#     #print(param_values)
#
#
#     for interaction in interactions:
#         for param_value in [0.2]:
#             key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', param_value)
#
#             if key_to_retrieve not in loaded_results:
#                 continue
#
#             value = loaded_results[key_to_retrieve]
#             alt_dict_value = value[1]
#             #print(alt_dict_value)
#
#             all_y_values.extend([np.sum(entry[1]) for scenario in scenarios for entry in alt_dict_value[scenario]])
#             all_y_percent_values.extend(
#                 [max(entry[1]) / np.sum(entry[1]) * 100 for scenario in scenarios for entry in alt_dict_value[scenario]])
#
#
#     #print(all_y_values)
#     y_min = min(all_y_values) - 0.1
#     y_max = max(all_y_values) + 0.1
#
#     y_percent_min = min(all_y_percent_values) - 2
#     y_percent_max = max(all_y_percent_values) + 2
#
#     # Create subplots
#     fig, axs = plt.subplots(2, 3, figsize=(20, 12))
#
#     # Loop over each interaction option for the given allowance
#     for interaction in interactions:
#         for param_value in [0.2]:
#             key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', param_value)
#
#             if key_to_retrieve not in loaded_results:
#                 print("We are getting stuck here")
#                 continue
#
#             value = loaded_results[key_to_retrieve]
#             alt_dict_value = value[1]
#             #print(f"This is the value for parameter {param_value}", alt_dict_value)
#
#             # Plot data for each scenario
#             for i, scenario in enumerate(scenarios):
#                 # For sum
#                 x = [entry[0] for entry in alt_dict_value[scenario]]
#                 y = [np.sum(entry[1]) for entry in alt_dict_value[scenario]]
#                 #print("What is y?", y)
#                 axs[0, i].plot(x, y, '-o', label=interaction)
#                 axs[0, i].set_title(f"Sum - {scenario}")
#                 axs[0, i].set_xlabel('Parameter Value')
#                 axs[0, i].set_ylabel('Sum of Metrics')
#                 axs[0, i].set_ylim(y_min, y_max)
#
#
#                 # For % contribution
#                 y_percent = [max(entry[1]) / np.sum(entry[1]) * 100 for entry in alt_dict_value[scenario]]
#                 axs[1, i].plot(x, y_percent, '-o', label=interaction)
#                 axs[1, i].set_title(f"% Contribution - {scenario}")
#                 axs[1, i].set_xlabel('Parameter Value')
#                 axs[1, i].set_ylabel('% Contribution of Largest Value')
#                 axs[1, i].set_ylim(y_percent_min, y_percent_max)
#     for i in range(3):
#         axs[0, i].legend()
#         axs[1, i].legend()
#     plt.subplots_adjust(top=0.8)
#     plt.tight_layout()
#     plt.suptitle(f"Allowance: {allowance}", fontsize=16, y=1)
#     #plt.show()
#     return fig
#

def plot_interaction(interaction):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#9467BD', '#8C564B', '#E377C2', '#7F7F7F'])

    with open('simulation_results_42.pkl', 'rb') as f:
        loaded_results = pickle.load(f)

    # Provided options
    allowances = ["strict_euclidean", "cosine", "size_cosine", "mean_euclidean"]
    scenarios = ['natural', 'low-removal', 'high-removal']

    # Determine the global y-axis limits for sum
    all_y_values = []

    # Determine the global y-axis limits for % contribution
    all_y_percent_values = []

    for allowance in allowances:

        key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)
        #print(key_to_retrieve)
        if key_to_retrieve not in loaded_results:
            print("We are getting stuck here")
            continue

        value = loaded_results[key_to_retrieve]
        alt_dict_value = value[1]
        #print("This is the item", allowance, alt_dict_value)

        all_y_values.extend([np.sum(entry[1]) for scenario in scenarios for entry in alt_dict_value[scenario]])
        all_y_percent_values.extend(
            [max(entry[1]) / np.sum(entry[1]) * 100 for scenario in scenarios for entry in alt_dict_value[scenario]])
        #(all_y_values)

    y_min = min(all_y_values) - 0.1


    y_max = max(all_y_values) + 0.1

    y_percent_min = min(all_y_percent_values) - 2
    y_percent_max = max(all_y_percent_values) + 2

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Loop over each allowance option for the given interaction
    for allowance in allowances:
        for param_value in [0.2]:
            key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', param_value)

            if key_to_retrieve not in loaded_results:
                print("We are getting stuck here")
                continue

            value = loaded_results[key_to_retrieve]
            alt_dict_value = value[1]

            # Plot data for each scenario
            for i, scenario in enumerate(scenarios):
                # For sum
                x = [entry[0] for entry in alt_dict_value[scenario]]
                y = [np.sum(entry[1]) for entry in alt_dict_value[scenario]]
                axs[0, i].plot(x, y, '-o', label=allowance)
                axs[0, i].set_title(f"Sum - {scenario}")
                axs[0, i].set_xlabel('Parameter Value')
                axs[0, i].set_ylabel('Sum of Metrics')
                axs[0, i].set_ylim(y_min, y_max)

                # For % contribution
                y_percent = [max(entry[1]) / np.sum(entry[1]) * 100 for entry in alt_dict_value[scenario]]
                axs[1, i].plot(x, y_percent, '-o', label=allowance)
                axs[1, i].set_title(f"% Contribution - {scenario}")
                axs[1, i].set_xlabel('Parameter Value')
                axs[1, i].set_ylabel('% Contribution of Largest Value')
                axs[1, i].set_ylim(y_percent_min, y_percent_max)

    for i in range(3):
        axs[0, i].legend()
        axs[1, i].legend()
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.suptitle(f"Interaction: {interaction}", fontsize=16, y=1)
    #plt.show()
    return fig

def plot_net_difference2(interaction):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#9467BD', '#8C564B', '#E377C2', '#7F7F7F'])

    with open('simulation_results_42.pkl', 'rb') as f:
        loaded_results = pickle.load(f)

    # Provided options
    allowances = ["strict_euclidean", "cosine", "mean_euclidean", "size_cosine"]
    scenarios = ['low-removal', 'high-removal']


    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Determine the global y-axis limits for sum and % contribution
    all_y_sum_diff_values = []
    all_y_percent_diff_values = []

    for allowance in allowances:

        key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

        if key_to_retrieve not in loaded_results:
            continue

        value = loaded_results[key_to_retrieve]
        alt_dict_value = value[1]

        natural_values = alt_dict_value['natural']

        for scenario in scenarios:
            scenario_values = alt_dict_value[scenario]

            # For sum differences
            sum_diffs = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                         zip(scenario_values, natural_values)]
            all_y_sum_diff_values.extend(sum_diffs)

            # For % contribution differences
            percent_diffs = [
                max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                entry, natural_entry in zip(scenario_values, natural_values)]
            all_y_percent_diff_values.extend(percent_diffs)

    y_sum_diff_min = min(all_y_sum_diff_values) - 0.1
    y_sum_diff_max = max(all_y_sum_diff_values) + 0.1

    y_percent_diff_min = min(all_y_percent_diff_values) - 5
    y_percent_diff_max = max(all_y_percent_diff_values) + 5

    # Loop over each interaction option
    for allowance in allowances:
        #for allowance in ["strict_euclidean", "cosine"]:
        key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

        if key_to_retrieve not in loaded_results:
            continue

        value = loaded_results[key_to_retrieve]
        alt_dict_value = value[1]

        natural_values = alt_dict_value['natural']

        # Plot data for each scenario
        for i, scenario in enumerate(scenarios):
            scenario_values = alt_dict_value[scenario]

            # For sum differences
            x = [entry[0] for entry in scenario_values]
            y_diff = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                      zip(scenario_values, natural_values)]
            axs[0, i].plot(x, y_diff, '-o', label=f"{interaction} - {allowance}")
            axs[0, i].set_title(f"Sum Difference - {scenario}")
            axs[0, i].set_xlabel('Parameter Value')
            axs[0, i].set_ylabel('Net Sum Difference')
            axs[0, i].set_ylim(y_sum_diff_min, y_sum_diff_max)
            axs[0, i].legend()

            # For % contribution differences
            y_percent_diff = [
                max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                entry, natural_entry in zip(scenario_values, natural_values)]
            axs[1, i].plot(x, y_percent_diff, '-o', label=f"{interaction} - {allowance}")
            axs[1, i].set_title(f"% Contribution Difference - {scenario}")
            axs[1, i].set_xlabel('Parameter Value')
            axs[1, i].set_ylabel('Net % Contribution Difference')
            axs[1, i].set_ylim(y_percent_diff_min, y_percent_diff_max)
            axs[1, i].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.suptitle(f"Interaction: {interaction}", fontsize=16, y=1)
    #plt.show()
    #plt.show()
    return fig

# Example usage:


options = ["strict_euclidean", "cosine", "mean_euclidean", "size_cosine"]

import os

def list_pickle_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file() and entry.name.endswith('.pkl')]

pickle_files = list_pickle_files('pickle_jar')
print(pickle_files)

def list_pickle_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file() and entry.name.endswith('.pkl')]

pickle_files = list_pickle_files('pickle_jar')

all_results = []

# Load results from multiple pickle files
for filename in pickle_files:
    with open(os.path.join('pickle_jar', filename), 'rb') as f:
        all_results.append(pickle.load(f))

plot_allowance("strict_euclidean", all_results)




# for option in options:
#     print(option)
#     fig1= plot_allowance(option)
#     fig2 = plot_net_difference(option)
#     fig1.savefig(f"images/{option}_allowance.png")
#     fig2.savefig(f"images/{option}_allowance_diff.png")
#
# options2 = ["bounded", "ensemble", "softmax", "sequential"]
# for option in options2:
#     fig3 = plot_interaction(option)
#     fig4 = plot_net_difference2(option)
#     fig3.savefig(f"images/{option}_interaction.png")
#     fig4.savefig(f"images/{option}_interaction_difference.png")

#plot_interaction("ensemble")