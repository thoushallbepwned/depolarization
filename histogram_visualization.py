import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results(pickle_files):
    all_results = []
    for file in pickle_files:
        with open(file, 'rb') as f:
            all_results.append(pickle.load(f))
    return all_results

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


result1 = all_results[0]
print("this is the length of result1",len(result1))
print(type(result1))
print(result1.keys())
subresult1 = result1[('noiseless', 'sequential', 'strict_euclidean', 'mixed', 0.8)]
print(type(subresult1))
#print(subresult1[0])
print(type(subresult1[0]))
print(subresult1[0].keys())
dataset = subresult1[0]['natural']
print(type(dataset))
print(dataset.keys())
before_data = dataset['df_before']
after_data = dataset['df_after']

print(type(before_data))

#print(before_data.shape())
print(before_data)

import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmaps(data_after, epsilon_list, allowances, interaction, mode):
    bin_count = 80
    global_min = data_after.min().min()
    global_max = data_after.max().max()
    bin_edges = np.linspace(global_min, global_max, bin_count + 1)

    fig, axes = plt.subplots(len(allowances), len(data_after.columns), figsize=(20, 5 * len(allowances)))

    for a_idx, allowance in enumerate(allowances):
        for dim_idx, col in enumerate(data_after.columns):
            heatmap_data = np.zeros((len(epsilon_list), bin_count))
            local_vmax = 0  # to store the max value for this specific heatmap
            for e_idx, epsilon in enumerate(epsilon_list):
                subresult = result1[('noiseless', interaction, allowance, 'mixed', epsilon)]
                dataset = subresult[0][mode]
                data = dataset['df_after']

                hist, _ = np.histogram(data[col], bins=bin_edges)
                heatmap_data[e_idx, :] = hist
                if hist.max() > local_vmax:
                    local_vmax = hist.max()

            sns.heatmap(heatmap_data, ax=axes[a_idx, dim_idx], cmap='viridis', vmin=0, vmax=local_vmax)
            axes[a_idx, dim_idx].set_title(f"{col} | Allowance: {allowance}")
            axes[a_idx, dim_idx].set_xlabel('Opinion Value')

            # Opinion axis ticks
            xticks_positions = np.linspace(0, bin_count, 5)
            axes[a_idx, dim_idx].set_xticks(xticks_positions)
            axes[a_idx, dim_idx].set_xticklabels([-1, -0.5, 0, 0.5, 1])

            # Epsilon axis ticks - every other tick
            axes[a_idx, dim_idx].set_yticks(np.arange(0, len(epsilon_list), 2))
            axes[a_idx, dim_idx].set_yticklabels(epsilon_list[::2])
            if dim_idx == 0:
                axes[a_idx, dim_idx].set_ylabel('Epsilon')

    fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.1, wspace=0.4, hspace=0.5)
    fig.suptitle(f"Opinion distribution under {interaction} and differing allowances, mode:{mode}", fontsize=16)
    plt.show()
    return fig
def plot_heatmaps_inv(data_after, epsilon_list, allowance, interactions, mode):
    bin_count = 80
    global_min = data_after.min().min()
    global_max = data_after.max().max()
    bin_edges = np.linspace(global_min, global_max, bin_count + 1)

    fig, axes = plt.subplots(len(interactions), len(data_after.columns), figsize=(20, 5 * len(interactions)))

    for int_idx, interaction in enumerate(interactions):
        for dim_idx, col in enumerate(data_after.columns):
            heatmap_data = np.zeros((len(epsilon_list), bin_count))
            local_vmax = 0  # to store the max value for this specific heatmap
            for e_idx, epsilon in enumerate(epsilon_list):
                subresult = result1[('noiseless', interaction, allowance, 'mixed', epsilon)]

                dataset = subresult[0][mode]
                data = dataset['df_after']

                hist, _ = np.histogram(data[col], bins=bin_edges)
                heatmap_data[e_idx, :] = hist
                if hist.max() > local_vmax:
                    local_vmax = hist.max()

            sns.heatmap(heatmap_data, ax=axes[int_idx, dim_idx], cmap='viridis', vmin=0, vmax=local_vmax)
            axes[int_idx, dim_idx].set_title(f"{col} | Interaction: {interaction}")
            axes[int_idx, dim_idx].set_xlabel('Opinion Value')

            # Opinion axis ticks
            xticks_positions = np.linspace(0, bin_count, 5)
            axes[int_idx, dim_idx].set_xticks(xticks_positions)
            axes[int_idx, dim_idx].set_xticklabels([-1, -0.5, 0, 0.5, 1])

            # Epsilon axis ticks - every other tick
            axes[int_idx, dim_idx].set_yticks(np.arange(0, len(epsilon_list), 2))
            axes[int_idx, dim_idx].set_yticklabels(epsilon_list[::2])
            if dim_idx == 0:
                axes[int_idx, dim_idx].set_ylabel('Epsilon')

        fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.1, wspace=0.4, hspace=0.5)
        fig.suptitle(f"Opinion distribution under different interactions with {allowance} allowance, mode:{mode}", fontsize=16)

    return fig
def compute_difference_histogram(data1, data2, bin_edges):
    hist1, _ = np.histogram(data1, bins=bin_edges)
    hist2, _ = np.histogram(data2, bins=bin_edges)
    return hist1 - hist2

def plot_heatmaps_difference(data_after, epsilon_list, allowances, interaction, removal_mode):
    bin_count = 80
    global_min = data_after.min().min()
    global_max = data_after.max().max()
    bin_edges = np.linspace(global_min, global_max, bin_count + 1)

    fig, axes = plt.subplots(len(allowances), len(data_after.columns), figsize=(20, 5 * len(allowances)))

    for a_idx, allowance in enumerate(allowances):
        for dim_idx, col in enumerate(data_after.columns):
            heatmap_data = np.zeros((len(epsilon_list), bin_count))
            for e_idx, epsilon in enumerate(epsilon_list):
                subresult = result1[('noiseless', interaction, allowance, 'mixed', epsilon)]

                data_natural = subresult[0]['natural']['df_after'][col]
                data_removal = subresult[0][removal_mode]['df_after'][col]

                diff_hist = compute_difference_histogram(data_removal,data_natural, bin_edges)

                heatmap_data[e_idx, :] = diff_hist

            sns.heatmap(heatmap_data, ax=axes[a_idx, dim_idx], cmap='bwr',
                        center=0)  # Using blue-white-red color map centered at 0
            axes[a_idx, dim_idx].set_title(f"{col} | Allowance: {allowance}")
            axes[a_idx, dim_idx].set_xlabel('Opinion Value')

            # Opinion axis ticks
            xticks_positions = np.linspace(0, bin_count, 5)
            axes[a_idx, dim_idx].set_xticks(xticks_positions)
            axes[a_idx, dim_idx].set_xticklabels([-1, -0.5, 0, 0.5, 1])

            # Epsilon axis ticks - every other tick
            axes[a_idx, dim_idx].set_yticks(np.arange(0, len(epsilon_list), 2))
            axes[a_idx, dim_idx].set_yticklabels(epsilon_list[::2])
            if dim_idx == 0:
                axes[a_idx, dim_idx].set_ylabel('Epsilon')

        fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.1, wspace=0.4, hspace=0.5)
        fig.suptitle(f"Density change for {interaction} interaction under {removal_mode}",
                     fontsize=16)
    plt.show()
    return fig

epsilon_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
allowances = ['mean_euclidean', 'strict_euclidean', 'cosine']
interactions = ["softmax", "sequential", "ensemble", "bounded"]
modes = ["natural", "high-removal", "low-removal"]

for interaction in interactions:
    # Plotting for natural mode
    #fig = plot_heatmaps(after_data, epsilon_list, allowances, interaction, "natural")
    #fig.savefig(f'final_figures/natural_heatmap_{interaction}.png', dpi=300, bbox_inches='tight')
    #
    # # Plotting difference histograms for high-removal mode
    fig = plot_heatmaps_difference(after_data, epsilon_list, allowances, interaction, "high-removal")
    fig.savefig(f'final_figures/difference_high-removal_heatmap_{interaction}.png', dpi=300, bbox_inches='tight')
    #
    # # Plotting difference histograms for low-removal mode
    fig = plot_heatmaps_difference(after_data, epsilon_list, allowances, interaction, "low-removal")
    fig.savefig(f'final_figures/difference_low-removal_heatmap_{interaction}.png', dpi=300, bbox_inches='tight')

#for allowance in allowances:
    # Plotting for natural mode
    #fig = plot_heatmaps_inv(after_data, epsilon_list, allowance, interactions, "natural")
    #fig.savefig(f'final_figures/natural_heatmap_{allowance}.png', dpi=300, bbox_inches='tight')

    # Note: If you also want the inversed difference plots (where interactions vary), you would need to adjust the `plot_heatmaps_inv` function similarly to the way we adjusted `plot_heatmaps` and then call that new function here.
