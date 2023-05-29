import torch
torch_version = torch.__version__
from torch_geometric.datasets import Planetoid, TUDataset
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool


dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS', transform=T.NormalizeFeatures())
# Split the dataset into train and test sets
print(len(dataset))
train_dataset = dataset[:667]  # Use the first 540 graphs for training
test_dataset = dataset[668:890]   # Use the rest for testing
val_dataset = dataset[891:1113]   # Use the rest for testing
# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
# data=dataset[0]
# dataset = Planetoid(root='.', name="Pubmed")
data = dataset[0]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)


import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

#helper functions

def accuracy(pred, target):
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()

class GraphSAGE(torch.nn.Module):


  """GraphSAGE"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.sage1 = SAGEConv(dim_in, dim_h)
    self.sage2 = SAGEConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.00001,
                                      weight_decay=0.001)

  def forward(self, x, edge_index, batch):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    h_graph = global_mean_pool(h, batch)
    return h_graph, F.log_softmax(h_graph, dim=1)

  def fit(self, data, epochs):
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = self.optimizer

      train_losses = []
      train_accuracies = []
      val_losses = []
      val_accuracies = []
      test_losses = []
      test_accuracies = []

      self.train()
      for epoch in range(epochs + 1):
          self.train()
          total_loss = 0
          total_correct = 0
          val_loss = 0
          val_correct = 0

          for batch in train_loader:
              batch = batch.to(device)
              optimizer.zero_grad()
              _, out = self(batch.x, batch.edge_index, batch.batch)
              loss = criterion(out, batch.y)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()
              _, pred = out.max(dim=1)
              total_correct += int((pred == batch.y).sum())

          # Append the average loss and accuracy for this epoch
          train_losses.append(total_loss / len(train_loader))
          train_accuracies.append(total_correct / len(train_dataset))

          # Switch to evaluation mode
          self.eval()

          with torch.no_grad():
              for val_batch in val_loader:
                  val_batch = val_batch.to(device)
                  val_batch.y = val_batch.y.to(device)  # Move targets to the same device as inputs
                  _, val_out = self(val_batch.x, val_batch.edge_index, val_batch.batch)
                  loss = criterion(val_out, val_batch.y)
                  val_loss += loss.item()
                  _, val_pred = val_out.max(dim=1)
                  #print(val_pred, val_batch.y)
                  val_correct += int((val_pred == val_batch.y).sum())

          # Append the average loss and accuracy for this epoch
          val_losses.append(val_loss / len(val_loader))
          val_accuracies.append(val_correct / len(val_dataset))

          test_loss = 0
          test_correct = 0
          for test_batch in test_loader:
              test_batch = test_batch.to(device)
              test_batch.y = test_batch.y.to(device)
              _, test_out = self(test_batch.x, test_batch.edge_index, test_batch.batch)
              loss = criterion(test_out, test_batch.y)
              test_loss += loss.item()
              _, test_pred = test_out.max(dim=1)
              test_correct += int((test_pred == test_batch.y).sum())

          # Append the average loss and accuracy for this epoch
          test_losses.append(test_loss / len(test_loader))
          test_accuracies.append(test_correct / len(test_dataset))

          if epoch % 10 == 0:
              print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Accuracy: {total_correct / len(train_dataset)}")
              print(f"Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {val_correct / len(val_dataset)}")
              print(f"Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_correct / len(test_dataset)}")

      return train_losses, train_accuracies, val_losses, val_accuracies, test_losses, test_accuracies

""" Entering training loop"""

"""checking for cuda"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GraphSAGE(dataset.num_features, 16, dataset.num_classes).to(device)

print("starting training loop")

train_losses, train_accuracies, val_losses, val_accuracies, test_losses, test_accuracies = model.fit(data, 200)

plt.figure(figsize=(12, 6))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print("completed training")


#
# print("testing this")
# #
# # Print information about the dataset
# print(f'Dataset: {dataset}')
# print('-------------------')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of nodes: {data.x.shape[0]}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
#
# # Print information about the graph
# print(f'\nGraph:')
# print('------')
# print(f'Training nodes: {sum(data.train_mask).item()}')
# print(f'Evaluation nodes: {sum(data.val_mask).item()}')
# print(f'Test nodes: {sum(data.test_mask).item()}')
# print(f'Edges are directed: {data.is_directed()}')
# print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Graph has loops: {data.has_self_loops()}')
# #
#
# print(data)

from torch_geometric.loader import NeighborLoader
import torch_geometric.sampler.neighbor_sampler as neighbor_sampler
from torch_geometric.utils import to_networkx



# # Create batches with neighbor sampling
# train_loader = NeighborLoader(
#     data,
#     num_neighbors=[5, 10],
#     batch_size=16,
#     input_nodes=data.train_mask,
# )

# Print each subgraph
# for i, subgraph in enumerate(train_loader):
#     print(f'Subgraph {i}: {subgraph}')
#
# # Plot each subgraph
# fig = plt.figure(figsize=(16,16))
# for idx, (subdata, pos) in enumerate(zip(train_loader, [221, 222, 223, 224])):
#     G = to_networkx(subdata, to_undirected=True)
#     ax = fig.add_subplot(pos)
#     ax.set_title(f'Subgraph {idx}')
#     plt.axis('off')
#     nx.draw_networkx(G,
#                     pos=nx.spring_layout(G, seed=0),
#                     with_labels=True,
#                     node_size=200,
#                     node_color=subdata.y,
#                     cmap="cool",
#                     font_size=10
#                     )
# plt.show()

from torch_geometric.utils import degree
from collections import Counter

# def plot_degree(data):
#   # Get list of degrees for each node
#   degrees = degree(data.edge_index[0]).numpy()
#
#   # Count the number of nodes for each degree
#   numbers = Counter(degrees)
#
#   # Bar plot
#   fig, ax = plt.subplots(figsize=(18, 6))
#   ax.set_xlabel('Node degree')
#   ax.set_ylabel('Number of nodes')
#   plt.bar(numbers.keys(),
#           numbers.values(),
#           color='#0A047A')
#
# # Plot node degrees from the original graph
# plot_degree(data)
# plt.show()
#
# # Plot node degrees from the last subgraph
# plot_degree(subdata)
# plt.show()
#