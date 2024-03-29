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

dataset = SNAPDataset(root='./data', name='ego-facebook')
print(dataset)
# load the first graph in the dataset
data = dataset[1].to(device)
print(data.x)
#print(data.node_features)


# Add self loops and convert to undirected graph
data.edge_index = to_undirected(data.edge_index)

# create one-hot encoded node features
data.x = data.x

#print(data.x.shape)
# Create positive and negative links for training and testing
transform  = RandomLinkSplit(is_undirected= False)
train_data, val_data, test_data = transform(data)
#print(train_data, val_data, test_data)

print(train_data)

print("what is train_data.edge_index?", train_data.edge_index)
print("what is train_data.num_nodes?", train_data.num_nodes)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return x

"putting a placeholder"

entry_layer = 347
model = GraphSAGE(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.005)


def get_link_labels(edge_label_index, edge_label):
    return edge_label.to(device)


def train(data):
    model.train()

    # Get positive and negative edge indices
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = data.edge_label_index[:, data.edge_label == 0]

    optimizer.zero_grad()

    # Get embeddings for nodes in the training data
    z = model(data.x, data.edge_index)

    # Calculate loss
    link_logits = torch.cat(
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in pos_edge_index.t().tolist()] +
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()], dim=0)

    link_labels = get_link_labels(data.edge_label_index, data.edge_label)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    # Calculate accuracy
    predictions = (torch.sigmoid(link_logits) > 0.5).long()
    accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy



def test(data):
    model.eval()
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    link_logits = torch.cat(
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in pos_edge_index.t().tolist()] +
        [(z[edge[0]] * z[edge[1]]).sum(dim=-1).unsqueeze(0) for edge in neg_edge_index.t().tolist()], dim=0)

    link_labels = get_link_labels(data.edge_label_index, data.edge_label)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    predictions = (torch.sigmoid(link_logits) > 0.5).long()
    accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)

    return loss.item(), accuracy



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

epochs = 5000
record_every = int(epochs / 20)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
# Training loop
for epoch in tqdm(range(1, epochs + 1)):
    if epoch % record_every == 0:
        train_loss, train_accuracy = train(train_data)  # Pass training data
        train_losses.append(train_loss)  # Store train loss
        train_accuracies.append(train_accuracy)  # Store train accuracy

        val_loss, val_accuracy = test(val_data)  # Pass validation data
        val_losses.append(val_loss)  # Store validation loss
        val_accuracies.append(val_accuracy)  # Store validation accuracy

        test_loss, test_accuracy = test(test_data)  # Pass test data
        test_losses.append(test_loss)  # Store test loss
        test_accuracies.append(test_accuracy)  # Store test accuracy
        print(" ")
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, '
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