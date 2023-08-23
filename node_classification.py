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
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import os
import random
from torch_geometric.nn import GCNConv
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def sample_pairs(edge_index, num_nodes):
    # Positive pairs are just the existing edges
    pos_pairs = edge_index.t()

    # Existing edges as a set for O(1) lookup
    edge_set = set(tuple(i.tolist()) for i in pos_pairs)

    # Sample negative edges on-the-fly
    neg_pairs = []
    while len(neg_pairs) < len(pos_pairs):
        i, j = torch.randint(0, 2000, (2,))
        if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
            neg_pairs.append((i, j))

    neg_pairs = torch.tensor(neg_pairs).long()

    return pos_pairs, neg_pairs


def sample_pairs_for_batch(edge_index, batch, num_nodes):
    pos_pairs_list, neg_pairs_list = [], []

    # Loop over each unique graph in the batch
    for graph_id in batch.unique():
        # Mask for the nodes that belong to the current graph
        node_mask = (batch == graph_id)

        # Extract edge indices corresponding to the current graph
        graph_edge_index = edge_index[:, node_mask[edge_index[0]]]

        # Adjust the edge indices to local indexing
        offset = node_mask.nonzero(as_tuple=True)[0][0]
        graph_edge_index -= offset

        pos_pairs, neg_pairs = sample_pairs(graph_edge_index, 2000)
        pos_pairs_list.append(pos_pairs)
        neg_pairs_list.append(neg_pairs)

    return pos_pairs_list, neg_pairs_list


def compute_contrastive_loss(embeddings, pos_pairs, neg_pairs, margin=0.5):
    try:
        # Compute embeddings for nodes in the positive and negative pairs
        pos_u, pos_v = pos_pairs[:, 0], pos_pairs[:, 1]
        neg_u, neg_v = neg_pairs[:, 0], neg_pairs[:, 1]

        pos_distances = torch.sum((embeddings[pos_u] - embeddings[pos_v])**2, dim=1)
        neg_distances = torch.sum((embeddings[neg_u] - embeddings[neg_v])**2, dim=1)
        # print("neg_u max:", neg_u.max(), "min:", neg_u.min())
        # print("neg_v max:", neg_v.max(), "min:", neg_v.min())
        # print("embeddings shape:", embeddings.shape)
        # Using hinge loss
        loss = torch.mean(F.relu(pos_distances - neg_distances + margin))

    except RuntimeError as e:
        if "CUDA error: device-side assert triggered" in str(e):
            print("Error encountered", embeddings[neg_u].shape, embeddings[neg_v].shape, embeddings.shape)
            print("Error encountered in compute_contrastive_loss. Skipping this iteration.")
            return torch.tensor(0.0, device=embeddings.device)  # Return a dummy loss
        else:
            raise e  # If it's a different RuntimeError, raise it

    return loss


start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_list = []

graph_directory = "graphs/end_state_training"
graph_files = [os.path.join(graph_directory, file) for file in os.listdir(graph_directory) if file.endswith('.p')]


# Shuffle the files
random.shuffle(graph_files)

# Splitting the data: Let's say 70% training, 15% validation, 15% test
train_split = int(0.7 * len(graph_files))
val_split = int(0.15 * len(graph_files)) + train_split

train_files = graph_files[:train_split]
val_files = graph_files[train_split:val_split]
test_files = graph_files[val_split:]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for preamble pass: {elapsed_time:.2f} seconds")

def load_graphs(file_list):
    data_list = []
    for file in file_list:
        g = pickle.load(open(file, "rb"))

        opinion = nx.get_node_attributes(g, 'opinion')
        # Convert node attributes to PyTorch tensor
        for node, data in g.nodes(data=True):
            data['opinion'] = torch.tensor(data['opinion'])

        # Convert the graph into PyG Data object
        data_nd = from_networkx(g)

        data = data_nd.to(device)
        data.x = data.opinion

        # Add self loops and convert to undirected graph
        data.edge_index = to_undirected(data.edge_index).to(device)

        # create one-hot encoded node features
        data.x = data.x

        data_list.append(data_nd.to(device))

    return Batch.from_data_list(data_list)

start_time = time.time()
train_data = load_graphs(train_files)
print(train_data)
print("Features", train_data.num_features)
val_data = load_graphs(val_files)
test_data = load_graphs(test_files)

batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("what is the number of batches", len(train_loader))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for loading graphs: {elapsed_time:.2f} seconds")
# Define the GraphSAGE model

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


start_time = time.time()

epochs = 100
model = GCN(train_data.num_features, 8).to(device)  # Create a new instance of the model
#model.load_state_dict(torch.load('GraphSAGE_model_node_classifier.pth'))  # Load the saved weights
model.train()  # Set the model back to training mode
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for initializing: {elapsed_time:.2f} seconds")

def train(loader):
    model.train()
    total_loss = 0

    for data in loader:


        optimizer.zero_grad()
        # Forward pass to get embeddings
        embeddings = model(data.x, data.edge_index)

        # Positive and negative sample pairs can be generated based on neighborhood proximity
        pos_pairs_list, neg_pairs_list = sample_pairs_for_batch(data.edge_index, data.batch, 2000)


        # You can aggregate the loss for each graph in the batch or compute it per graph
        # Here, I'll aggregate for simplicity
        batch_loss = 0

        for pos_pairs, neg_pairs in zip(pos_pairs_list, neg_pairs_list):
            #start_time = time.time()
            loss = compute_contrastive_loss(embeddings, pos_pairs, neg_pairs)
            batch_loss += loss

            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"Time taken for batch pass: {elapsed_time:.2f} seconds")

        # Average the batch loss
        batch_loss = batch_loss / len(pos_pairs_list)

        # Backpropagation
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / len(loader)


def validate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            # Generate embeddings
            embeddings = model(data.x, data.edge_index)

            # Positive and negative sample pairs can be generated based on neighborhood proximity
            pos_pairs_list, neg_pairs_list = sample_pairs_for_batch(data.edge_index, data.batch, 2000)

            # Aggregating the loss for each graph in the batch
            batch_loss = 0
            for pos_pairs, neg_pairs in zip(pos_pairs_list, neg_pairs_list):
                loss = compute_contrastive_loss(embeddings, pos_pairs, neg_pairs)
                batch_loss += loss

            # Average the batch loss
            batch_loss = batch_loss / len(pos_pairs_list)

            total_loss += batch_loss.item()

    return total_loss / len(loader)


loss_values = []
best_val_loss = float('inf')
for epoch in tqdm(range(1, epochs + 1)):
    train_loss = train(train_loader)
    val_loss = validate(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model if needed
        torch.save(model.state_dict(), 'contrastive_node_classifier.pth')

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# After training, you can evaluate the model on the test set if needed
test_loss = validate(test_loader)
print(f'Test Loss: {test_loss:.4f}')



# Create a list to gather embeddings from all mini-batches
all_embeddings = []

model.eval()
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Processing batches"):
        embeddings_batch = model(batch.x, batch.edge_index)
        all_embeddings.append(embeddings_batch.cpu().numpy())

# Concatenate embeddings from all batches
embeddings = np.vstack(all_embeddings)

# Use KMeans to assign cluster labels
kmeans = KMeans(n_clusters=4).fit(embeddings)  # Example for 3 clusters
labels = kmeans.labels_

sil_score = silhouette_score(embeddings, labels)

print(f'Silhouette Score: {sil_score:.4f}')

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()

# Reduce dimensions to 3 using t-SNE
tsne = TSNE(n_components=3, init = "pca", random_state=42, learning_rate = 'auto')
embeddings_3d = tsne.fit_transform(embeddings)

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
#torch.save(model.state_dict(), 'GraphSAGE_model_node_classifier.pth')
