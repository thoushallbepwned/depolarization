import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import train_test_split_edges, to_undirected, negative_sampling


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = './data'
dataset_name = 'ego-Facebook'

# load the dataset
dataset = SNAPDataset(root=root_dir, name=dataset_name)

# load the first graph in the dataset
data = dataset[0].to(device)


# Add self loops and convert to undirected graph
data.edge_index = to_undirected(data.edge_index)

# create one-hot encoded node features
data.x = torch.eye(data.num_features, dtype=torch.float).to(device)

print(data.x.shape)
# Create positive and negative links for training and testing
data = train_test_split_edges(data)



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


print(data.num_features)

"putting a placeholder"

entry_layer = 347
model = GraphSAGE(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.cat([torch.ones(pos_edge_index.size(1), ),
                             torch.zeros(neg_edge_index.size(1), )], dim=0).to(device)
    return link_labels


def train():
    model.train()

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()

    # Get embeddings for nodes in the training data
    z = model(data.x, data.train_pos_edge_index)

    # Calculate loss
    link_logits = torch.cat(
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in data.train_pos_edge_index.t().tolist()] +
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()], dim=0)

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    # Calculate accuracy
    predictions = (torch.sigmoid(link_logits) > 0.5).long()
    accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy


def test():
    model.eval()
    losses = []
    accuracies = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        with torch.no_grad():
            # Get embeddings for nodes in the test data
            z = model(data.x, data.train_pos_edge_index)

        # Calculate loss
        link_logits = torch.cat(
            [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in pos_edge_index.t().tolist()] +
            [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()], dim=0)

        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        losses.append(loss.item())

        # Calculate accuracy
        predictions = (torch.sigmoid(link_logits) > 0.5).long()
        accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)
        accuracies.append(accuracy)

    return losses + accuracies


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

epochs = 100
record_every = int(epochs / 20)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in tqdm(range(1, epochs + 1)):
    if epoch % record_every == 0:
        loss, accuracy = train()
        train_losses.append(loss)  # Store train loss
        train_accuracies.append(accuracy)  # Store train accuracy

        val_loss, val_accuracy, test_loss, test_accuracy = test()
        val_losses.append(val_loss)  # Store validation loss
        val_accuracies.append(val_accuracy)  # Store validation accuracy
        test_losses.append(test_loss)  # Store test loss
        test_accuracies.append(test_accuracy)  # Store test accuracy

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

# Plot losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(record_every, epochs + 1, record_every), train_losses, label='Training')
plt.plot(range(record_every, epochs + 1, record_every), val_losses, label='Validation')
plt.plot(range(record_every, epochs + 1, record_every), test_losses, label='Test')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(range(record_every, epochs + 1, record_every), train_accuracies, label='Training')
plt.plot(range(record_every, epochs + 1, record_every), val_accuracies, label='Validation')
plt.plot(range(record_every, epochs + 1, record_every), test_accuracies, label='Test')
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()