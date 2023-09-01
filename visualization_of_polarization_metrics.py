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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_allowance(allowance, all_results):
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Provided options
    interactions = ["sequential", "softmax", "bounded", "ensemble"]
    scenarios = ['natural', 'low-removal', 'high-removal']
    label_map = {
        "mean_euclidean": "L$_2$",
        "strict_euclidean": "L$_\infty$",
        "softmax": "proportional",
        "cosine": "cosine"
    }
    scale_factor = 0.5
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
            #var_sums = np.var(all_sums[scenario], axis=0)

            n_sums = len(all_sums[scenario])
            n_percents = len(all_percents[scenario])

            avg_percents = np.mean(all_percents[scenario], axis=0)
            #var_percents = np.var(all_percents[scenario], axis=0)

            sem_sums = np.std(all_sums[scenario], axis=0) / np.sqrt(n_sums)
            sem_percents = np.std(all_percents[scenario], axis=0) / np.sqrt(n_percents)

            all_y_values.append(avg_sums)
            all_y_percent_values.append(avg_percents)

            # Plot data for each scenario
            x = [entry[0] for entry in alt_dict_value[scenario]]

            axs[0, scenarios.index(scenario)].errorbar(x, avg_sums, yerr=np.sqrt(sem_sums)*scale_factor, fmt='-o',
                                                       label=label_map.get(interaction, interaction))
            axs[0, scenarios.index(scenario)].set_title(f"Intervention mode - {scenario}")
            #axs[0, scenarios.index(scenario)].set_xlabel('Epsilon Value')
            axs[0, scenarios.index(scenario)].set_ylabel('Summmed fragmentation')

            axs[1, scenarios.index(scenario)].errorbar(x, avg_percents, yerr=np.sqrt(sem_percents)*scale_factor, fmt='-o', label=label_map.get(interaction, interaction))

            #axs[1, scenarios.index(scenario)].set_title(f"Intervention mode - {scenario}")
            axs[1, scenarios.index(scenario)].set_xlabel('Epsilon Value')
            axs[1, scenarios.index(scenario)].set_ylabel('Principal component fragmentation (PCF) in %')

    for i in range(3):
        axs[0, i].legend()
        axs[1, i].legend()

    flattened_y_values = np.array(all_y_values).flatten()
    y_min = 0.3
    y_max = 1.3
    y_percent_min = 55
    y_percent_max = 80
    #y_min = min(flattened_y_values) - 0.1
    #y_max = max(flattened_y_values) + 0.1
    flat_y_perc = np.array(all_y_percent_values).flatten()
    #y_percent_min = min(flat_y_perc) - 2
    #y_percent_max = max(flat_y_perc) + 2

    for i in range(3):
        axs[0, i].set_ylim(y_min, y_max)
        axs[1, i].set_ylim(y_percent_min, y_percent_max)


    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0, rect=None)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Allowance: {label_map.get(allowance, allowance)}", fontsize=24, y=0.98)

    plt.show()
    return fig


def plot_interaction(interaction, all_results):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#9467BD', '#8C564B', '#E377C2', '#7F7F7F'])

    # Provided options
    allowances = ["strict_euclidean", "cosine", "mean_euclidean"]
    scenarios = ['natural', 'low-removal', 'high-removal']

    # Map to renamed labels
    label_map = {
        "mean_euclidean": "L$_2$",
        "strict_euclidean": "L$_\infty$",
        "softmax": "proportional",
        "cosine": "cosine"
    }
    scale_factor = 0.5
    # Determine the global y-axis limits for sum
    all_y_values = []
    # Determine the global y-axis limits for % contribution
    all_y_percent_values = []

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    for allowance in allowances:
        all_sums = {scenario: [] for scenario in scenarios}
        all_percents = {scenario: [] for scenario in scenarios}

        for loaded_results in all_results:
            key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)
            if key_to_retrieve not in loaded_results:
                print("are we getting stuck here?")
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
            n_sums = len(all_sums[scenario])
            n_percents = len(all_percents[scenario])

            sem_sums = np.std(all_sums[scenario], axis=0) / np.sqrt(n_sums)
            sem_percents = np.std(all_percents[scenario], axis=0) / np.sqrt(n_percents)

            avg_percents = np.mean(all_percents[scenario], axis=0)

            all_y_values.append(avg_sums)
            all_y_percent_values.append(avg_percents)

            # Plot data for each scenario
            x = [entry[0] for entry in alt_dict_value[scenario]]

            # Use the label_map to get the correct label name
            axs[0, scenarios.index(scenario)].errorbar(x, avg_sums, yerr=np.sqrt(sem_sums)*scale_factor, fmt='-o', label=label_map.get(allowance, allowance))
            axs[0, scenarios.index(scenario)].set_title(f"Intervention mode - {scenario}")
            # axs[0, scenarios.index(scenario)].set_xlabel('Epsilon Value')
            axs[0, scenarios.index(scenario)].set_ylabel('Summmed fragmentation')

            axs[1, scenarios.index(scenario)].errorbar(x, avg_percents, yerr=np.sqrt(sem_percents)*scale_factor, fmt='-o', label=label_map.get(allowance, allowance))
            axs[1, scenarios.index(scenario)].set_xlabel('Epsilon Value')
            axs[1, scenarios.index(scenario)].set_ylabel('Principal component fragmentation (PCF) in %')

    for i in range(3):
        axs[0, i].legend()
        axs[1, i].legend()

    y_min = 0.3
    y_max = 1.3
    y_percent_min = 55
    y_percent_max = 80

    for i in range(3):
        axs[0, i].set_ylim(y_min, y_max)
        axs[1, i].set_ylim(y_percent_min, y_percent_max)

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0, rect=None)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Interaction: {label_map.get(interaction, interaction)}", fontsize=24, y=0.98)

    plt.show()
    return fig

def plot_net_difference_average(allowance, all_results):
    mpl.rcParams.update(mpl.rcParamsDefault)

    interactions = ["sequential", "softmax", "bounded", "ensemble"]
    scenarios = ['low-removal', 'high-removal']
    label_map = {
        "mean_euclidean": "L$_2$",
        "strict_euclidean": "L$_\infty$",
        "softmax": "proportional",
        "cosine": "cosine"
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    all_y_sum_diff_values = []
    all_y_percent_diff_values = []

    avg_diffs = {}
    avg_percent_diffs = {}
    var_diffs = {}
    var_percent_diffs = {}

    x_values = np.arange(0.15, 0.85 + 0.05, 0.05)  # Values from 0.15 to 0.85 with an interval of 0.05
    x_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for interaction in interactions:
        avg_diffs[interaction] = {}
        avg_percent_diffs[interaction] = {}
        var_diffs[interaction] = {}
        var_percent_diffs[interaction] = {}

        for scenario in scenarios:
            sum_diff_vals = []
            percent_diff_vals = []
            for loaded_results in all_results:
                key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

                if key_to_retrieve not in loaded_results:
                    continue

                value = loaded_results[key_to_retrieve]
                alt_dict_value = value[1]
                natural_values = alt_dict_value['natural']
                scenario_values = alt_dict_value[scenario]

                # For sum differences
                sum_diffs = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                             zip(scenario_values, natural_values)]
                sum_diff_vals.append(sum_diffs)

                # For % contribution differences
                percent_diffs = [
                    max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                    entry, natural_entry in zip(scenario_values, natural_values)]
                percent_diff_vals.append(percent_diffs)

            avg_diffs[interaction][scenario] = np.mean(sum_diff_vals, axis=0)
            avg_percent_diffs[interaction][scenario] = np.mean(percent_diff_vals, axis=0)
            var_diffs[interaction][scenario] = np.var(sum_diff_vals, axis=0)
            var_percent_diffs[interaction][scenario] = np.var(percent_diff_vals, axis=0)

            all_y_sum_diff_values.extend(avg_diffs[interaction][scenario])
            all_y_percent_diff_values.extend(avg_percent_diffs[interaction][scenario])

    y_sum_diff_min = min(all_y_sum_diff_values) - 0.1
    y_sum_diff_max = max(all_y_sum_diff_values) + 0.1
    y_percent_diff_min = min(all_y_percent_diff_values) - 5
    y_percent_diff_max = max(all_y_percent_diff_values) + 5

    for interaction in interactions:
        for i, scenario in enumerate(scenarios):
            #x = range(len(avg_diffs[interaction][scenario]))  # Assuming x-values are indices
            x = x_values[:len(avg_diffs[interaction][scenario])]
            # For sum differences
            label = f"{label_map.get(interaction, interaction)}"
            axs[0, i].errorbar(x, avg_diffs[interaction][scenario], yerr=np.sqrt(var_diffs[interaction][scenario]),
                               fmt='-o', label=label)
            axs[0, i].set_title(f"Intervention - {scenario}")
            axs[0, i].set_xlabel('Epsilon Value')
            axs[0, i].set_xticks(x_ticks)  # Set x-axis ticks
            axs[0, i].set_ylabel('$\Delta$ Summed Fragmentation')
            axs[0, i].set_ylim(y_sum_diff_min, y_sum_diff_max)
            axs[0, i].legend()

            # For % contribution differences
            axs[1, i].errorbar(x, avg_percent_diffs[interaction][scenario],
                               yerr=np.sqrt(var_percent_diffs[interaction][scenario]),
                               fmt='-o', label=label)
            axs[1, i].set_xlabel('Epsilon Value')
            axs[1, i].set_xticks(x_ticks)  # Set x-axis ticks
            axs[1, i].set_ylabel('$\Delta$ % PCF')
            axs[1, i].set_ylim(y_percent_diff_min, y_percent_diff_max)
            axs[1, i].legend()

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0, rect=None)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Allowance: {label_map.get(allowance, allowance)}", fontsize=24, y=0.98)
    plt.show()
    return fig
def plot_net_difference2_average(interaction, all_results):
    mpl.rcParams.update(mpl.rcParamsDefault)

    allowances = ["strict_euclidean", "cosine", "mean_euclidean"]
    scenarios = ['low-removal', 'high-removal']
    label_map = {
        "mean_euclidean": "L$_2$",
        "strict_euclidean": "L$_\infty$",
        "softmax": "proportional",
        "cosine": "cosine"
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    all_y_sum_diff_values = []
    all_y_percent_diff_values = []

    avg_diffs = {}
    avg_percent_diffs = {}
    var_diffs = {}
    var_percent_diffs = {}

    for allowance in allowances:
        avg_diffs[allowance] = {}
        avg_percent_diffs[allowance] = {}
        var_diffs[allowance] = {}
        var_percent_diffs[allowance] = {}

        for scenario in scenarios:
            sum_diff_vals = []
            percent_diff_vals = []
            for loaded_results in all_results:
                key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

                if key_to_retrieve not in loaded_results:
                    continue

                value = loaded_results[key_to_retrieve]
                alt_dict_value = value[1]
                natural_values = alt_dict_value['natural']
                scenario_values = alt_dict_value[scenario]

                # For sum differences
                sum_diffs = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in
                             zip(scenario_values, natural_values)]
                sum_diff_vals.append(sum_diffs)

                # For % contribution differences
                percent_diffs = [
                    max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for
                    entry, natural_entry in zip(scenario_values, natural_values)]
                percent_diff_vals.append(percent_diffs)

            n = len(sum_diff_vals)
            avg_diffs[allowance][scenario] = np.mean(sum_diff_vals, axis=0)
            avg_percent_diffs[allowance][scenario] = np.mean(percent_diff_vals, axis=0)
            var_diffs[allowance][scenario] = np.var(sum_diff_vals, axis=0)
            var_percent_diffs[allowance][scenario] = np.var(percent_diff_vals, axis=0)
            sem_diffs = np.sqrt(var_diffs[allowance][scenario]) / np.sqrt(n)
            sem_percent_diffs = np.sqrt(var_percent_diffs[allowance][scenario]) / np.sqrt(n)

            all_y_sum_diff_values.extend(avg_diffs[allowance][scenario])
            all_y_percent_diff_values.extend(avg_percent_diffs[allowance][scenario])

    y_sum_diff_min = min(all_y_sum_diff_values) - 0.1
    y_sum_diff_max = max(all_y_sum_diff_values) + 0.1
    y_percent_diff_min = min(all_y_percent_diff_values) - 5
    y_percent_diff_max = max(all_y_percent_diff_values) + 5

    # Specify new x-axis values and their respective tick labels
    x_values = np.arange(0.15, 0.85 + 0.05, 0.05)  # Values from 0.15 to 0.85 with an interval of 0.05
    x_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Ticks at specified values

    for allowance in allowances:
        for i, scenario in enumerate(scenarios):
            # Instead of using range for x, use the predefined x_values
            x = x_values[:len(avg_diffs[allowance][scenario])]

            # For sum differences
            label = f"{label_map.get(allowance, allowance)}"
            axs[0, i].errorbar(x, avg_diffs[allowance][scenario], yerr=sem_diffs,
                               fmt='-o', label=label)
            axs[0, i].set_title(f"Intervention - {scenario}")
            #axs[0, i].set_xlabel('Epsilon Value')
            axs[0, i].set_xticks(x_ticks)  # Set x-axis ticks
            axs[0, i].set_ylabel('$\Delta$ Summed Fragmentation')
            axs[0, i].set_ylim(y_sum_diff_min, y_sum_diff_max)
            axs[0, i].legend()

            # For % contribution differences
            axs[1, i].errorbar(x, avg_percent_diffs[allowance][scenario],
                               yerr=sem_percent_diffs,
                               fmt='-o', label=label)
            axs[1, i].set_xlabel('Epsilon Value')
            axs[1, i].set_xticks(x_ticks)  # Set x-axis ticks
            axs[1, i].set_ylabel('$\Delta$ % PCF')
            axs[1, i].set_ylim(y_percent_diff_min, y_percent_diff_max)
            axs[1, i].legend()

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0, rect=None)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Interaction: {label_map.get(interaction, interaction)}", fontsize=24, y=0.98)
    plt.show()
    return fig


# Example usage:

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



options = ["strict_euclidean", "cosine", "mean_euclidean"]
for option in options:
    fig1 = plot_allowance(option, all_results)
    fig2 = plot_net_difference_average(option, all_results)
    fig1.savefig(f"final_figures/allowance_{option}.png")
    fig2.savefig(f"final_figures/allowance_{option}_diff.png")

interactions = ["sequential", "softmax", "bounded", "ensemble"]

for interaction in interactions:
    fig3 = plot_interaction(interaction, all_results)
    fig4 = plot_net_difference2_average(interaction, all_results)
    fig3.savefig(f"final_figures/interaction_{interaction}.png")
    fig4.savefig(f"final_figures/interaction_{interaction}_diff.png")
