import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import from_networkx
import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

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


"Testing the inference on a loaded dadaset"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

version = "10000_nodes"

file = f"graphs/2000_nodes/final_graph_ensemble_mean_euclidean_mixed_0.65_noisy.p"
#g = pickle.load(open("final_graph_softmax_mean_euclidean_polarized.p", "rb"))
g = pickle.load(open(file, "rb"))

#final_opinions = {node: opinion for node, opinion in zip(g.nodes(), opinions)}

opinions = nx.get_node_attributes(g, 'opinion')
# Convert node attributes to PyTorch tensor
for node, data in g.nodes(data=True):
    data['opinion'] = torch.tensor(data['opinion'])

# Convert the graph into PyG Data object
data_nd = from_networkx(g)


data = data_nd.to(device)
data.x = data.opinion


# Initialize the model
loaded_model = GraphSAGE(data.num_features, 4).to(device)

# Load the state dict into the model
loaded_model.load_state_dict(torch.load('GraphSAGE_model_node_classifier.pth'))

# Set the model to evaluation mode
loaded_model.eval()

with torch.no_grad():
    new_embeddings = loaded_model(data.x, data.edge_index)

print(new_embeddings.shape)

print(new_embeddings)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Get the 4D embeddings
embeddings_np = new_embeddings.cpu().numpy()

# Assign dimensions 1, 2, 3, and 4
x = embeddings_np[:, 0]
y = embeddings_np[:, 1]
z = embeddings_np[:, 2]
color_dim = embeddings_np[:, 3]

# Create a colormap for the fourth dimension
color_map = cm.ScalarMappable(cmap=cm.jet)
colors = color_map.to_rgba(color_dim)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter data on the 3D plot with color representing the fourth dimension
scatter = ax.scatter(x, y, z, c=colors, cmap='jet', depthshade=False)
cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
ax.set_title('3D Visualization of new_embeddings with Color Representing Fourth Dimension')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Convert the opinions into a numpy array
opinions_np = np.array([data['opinion'].cpu().numpy() for _, data in g.nodes(data=True)])

# Check if the opinions are indeed 4D
if opinions_np.shape[1] != 4:
    print("The opinions are not 4D!")
    exit()

# Assign dimensions 1, 2, 3, and 4
x = opinions_np[:, 0]
y = opinions_np[:, 1]
z = opinions_np[:, 2]
color_dim = opinions_np[:, 3]

# Create a colormap for the fourth dimension
color_map = cm.ScalarMappable(cmap=cm.jet)
colors = color_map.to_rgba(color_dim)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter data on the 3D plot with color representing the fourth dimension
scatter = ax.scatter(x, y, z, c=colors, cmap='jet', depthshade=False)
cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')

ax.set_xlabel('Opinion Dim 1')
ax.set_ylabel('Opinion Dim 2')
ax.set_zlabel('Opinion Dim 3')
ax.set_title('3D Visualization of Node Opinions with Color Representing Fourth Dimension')
plt.show()

# Compute the difference between embeddings and opinions
diff = new_embeddings.cpu().numpy() - opinions_np

# Assign dimensions 1, 2, 3, and 4
x_diff = diff[:, 0]
y_diff = diff[:, 1]
z_diff = diff[:, 2]
color_diff_dim = diff[:, 3]

# Create a colormap for the fourth dimension of the difference
color_map_diff = cm.ScalarMappable(cmap=cm.jet)
colors_diff = color_map_diff.to_rgba(color_diff_dim)

# Create a 3D scatter plot for the difference
fig_diff = plt.figure(figsize=(10, 7))
ax_diff = fig_diff.add_subplot(111, projection='3d')

# Scatter data on the 3D plot with color representing the fourth dimension
scatter_diff = ax_diff.scatter(x_diff, y_diff, z_diff, c=colors_diff, cmap='jet', depthshade=False)
cbar_diff = fig_diff.colorbar(scatter_diff, ax=ax_diff, orientation='vertical')

ax_diff.set_xlabel('Difference Dim 1')
ax_diff.set_ylabel('Difference Dim 2')
ax_diff.set_zlabel('Difference Dim 3')
ax_diff.set_title('3D Visualization of Differences (Embeddings - Opinions) with Color Representing Fourth Dimension')
plt.show()

import numpy as np

def compute_similarity_ranking(matrix):
    """ Compute similarity ranking for each node."""
    rankings = []
    for i in range(matrix.shape[0]):
        dists = np.linalg.norm(matrix - matrix[i], axis=1)
        ranking = np.argsort(dists)
        rankings.append(ranking)
    return rankings

def get_top_n_pairs(rankings, n):
    """ Retrieve top-N pairs from rankings."""
    pairs = set()
    for i, ranking in enumerate(rankings):
        for j in range(1, n+1):  # Start from 1 because the 0th index would be the node itself
            pairs.add((i, ranking[j]))
    return pairs

embeddings_np = new_embeddings.cpu().numpy()
opinions_np = np.array([data['opinion'].numpy() for _, data in g.nodes(data=True)])

embedding_rankings = compute_similarity_ranking(embeddings_np)
opinion_rankings = compute_similarity_ranking(opinions_np)

N = 10  # Change this to your desired N
top_n_embedding_pairs = get_top_n_pairs(embedding_rankings, N)
top_n_opinion_pairs = get_top_n_pairs(opinion_rankings, N)

# Find common pairs
common_pairs = top_n_embedding_pairs.intersection(top_n_opinion_pairs)

total_possible_pairs_in_top_N = 2000 * N  # where N is the number of top pairs you're considering for each node
percentage_common = (len(common_pairs) / total_possible_pairs_in_top_N) * 100

print(f"Percentage of common pairs in top-N: {percentage_common:.2f}%")

expected_common_pairs = (N / 1999)**2 * N * 2000
print(f"Expected number of common pairs at random: {expected_common_pairs:.2f}")



print("Number of common pairs in top-N:", len(common_pairs))

expected_common_pairs = (N / 1999)**2 * N * 2000
expected_percentage = (expected_common_pairs / (N * 2000)) * 100

print(f"Expected number of common pairs at random: {expected_common_pairs:.2f}")
print(f"Expected percentage of common pairs at random: {expected_percentage:.2f}%")

