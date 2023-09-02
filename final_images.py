import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def load_all_results(pickle_files):
    return [pickle.load(open(file, 'rb')) for file in pickle_files]


def aggregate_results(all_results, key):
    return [results[key] for results in all_results if key in results]


def professionalize_plot(ax):
    """Helper function to give the plots a consistent and polished aesthetic."""
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.4)


def relabel_interaction(interaction):
    """Helper function to rename interactions as per user requirement."""
    interaction_map = {
        "sequential": "Sequential",
        "softmax": "Proportional",
        "bounded": "L2 Allowance",
        "ensemble": "L∞ Allowance"
    }
    return interaction_map.get(interaction, interaction)


def plot_net_difference_average(allowance, all_results):
    interactions = ["sequential", "softmax", "bounded", "ensemble"]
    scenarios = ['low-removal', 'high-removal']

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    all_y_sum_diff_values = []
    all_y_percent_diff_values = []

    data = {interaction: {scenario: {} for scenario in scenarios} for interaction in interactions}

    for interaction in interactions:
        for scenario in scenarios:
            sum_diff_vals, percent_diff_vals = [], []

            for loaded_results in all_results:
                key_to_retrieve = ('noiseless', interaction, allowance, 'mixed', 0.20)

                if key_to_retrieve not in loaded_results:
                    continue

                value = loaded_results[key_to_retrieve]
                alt_dict_value = value[1]
                natural_values = alt_dict_value['natural']
                scenario_values = alt_dict_value[scenario]

                sum_diffs = [np.sum(entry[1]) - np.sum(natural_entry[1]) for entry, natural_entry in zip(scenario_values, natural_values)]
                percent_diffs = [max(entry[1]) / np.sum(entry[1]) * 100 - max(natural_entry[1]) / np.sum(natural_entry[1]) * 100 for entry, natural_entry in zip(scenario_values, natural_values)]

                sum_diff_vals.append(sum_diffs)
                percent_diff_vals.append(percent_diffs)

            data[interaction][scenario] = {
                'avg_diffs': np.mean(sum_diff_vals, axis=0),
                'avg_percent_diffs': np.mean(percent_diff_vals, axis=0),
                'var_diffs': np.var(sum_diff_vals, axis=0),
                'var_percent_diffs': np.var(percent_diff_vals, axis=0)
            }

            all_y_sum_diff_values.extend(data[interaction][scenario]['avg_diffs'])
            all_y_percent_diff_values.extend(data[interaction][scenario]['avg_percent_diffs'])

    for i, scenario in enumerate(scenarios):
        for interaction in interactions:
            x = range(len(data[interaction][scenario]['avg_diffs']))

            # For sum differences
            axs[0, i].errorbar(x, data[interaction][scenario]['avg_diffs'], yerr=np.sqrt(data[interaction][scenario]['var_diffs']), fmt='-o', label=relabel_interaction(interaction))
            professionalize_plot(axs[0, i])
            axs[0, i].set_title(f"Sum Difference - {scenario.capitalize()}")
            axs[0, i].set_xlabel('Parameter Value')
            axs[0, i].set_ylabel('Net Sum Difference')

            # For % contribution differences
            axs[1, i].errorbar(x, data[interaction][scenario]['avg_percent_diffs'], yerr=np.sqrt(data[interaction][scenario]['var_percent_diffs']), fmt='-o', label=relabel_interaction(interaction))
            professionalize_plot(axs[1, i])
            axs[1, i].set_title(f"% Contribution Difference - {scenario.capitalize()}")
            axs[1, i].set_xlabel('Parameter Value')
            axs[1, i].set_ylabel('Net % Contribution Difference')

    fig.suptitle(f"Allowance: {allowance}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    return fig

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
    #fig1 = plot_allowance(option, all_results)
    plot_net_difference_average(option, all_results)
