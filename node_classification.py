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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

version = "10000_nodes"

file = f"graphs/2000_nodes/final_graph_sequential_mean_euclidean_mixed_0.65_noisy.p"
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

print("what is the data structure", data)
print(data.x)
#print(data.node_features)


# Add self loops and convert to undirected graph
data.edge_index = to_undirected(data.edge_index).to(device)

# create one-hot encoded node features
data.x = data.x


# Define the GraphSAGE model
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
"putting a placeholder"

#entry_layer = 347
model = GraphSAGE(data.num_features, 4).to(device)
print(data.num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

def train(data):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    adj_pred = torch.mm(z, z.t())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(adj_pred, torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(device), torch.Size([data.num_nodes, data.num_nodes])).to_dense())
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 200

loss_values = []

for epoch in tqdm(range(1, epochs + 1)):
    loss = train(data)
    loss_values.append(loss)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Obtain node embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Use KMeans to assign cluster labels
kmeans = KMeans(n_clusters=4).fit(embeddings.cpu().numpy())  # Example for 3 clusters
labels = kmeans.labels_

sil_score = silhouette_score(embeddings.cpu().numpy(), labels)

print(f'Silhouette Score: {sil_score:.4f}')

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()

# Reduce dimensions to 3 using t-SNE
tsne = TSNE(n_components=3)
embeddings_3d = tsne.fit_transform(embeddings.cpu().numpy())

# Create a 3D scatter plot
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter data on the 3D plot
for i in range(4):  # You have 8 clusters
    ax.scatter(embeddings_3d[labels == i, 0], embeddings_3d[labels == i, 1], embeddings_3d[labels == i, 2],
               label=f'Cluster {i}')

ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.set_zlabel('t-SNE dim 3')
ax.set_title('3D Clusters Visualization')
ax.legend()
plt.show()

# Saving the model
torch.save(model.state_dict(), 'GraphSAGE_model_node_classifier.pth')