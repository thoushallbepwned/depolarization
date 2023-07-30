import torch
from torch_geometric.datasets import TUDataset, SNAPDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges, to_undirected, from_networkx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle
import torch_geometric.utils as utils
import os

# Metric Calculation
def get_metrics(true_labels, pred_labels, probabilities):
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc_roc = roc_auc_score(true_labels, probabilities)

    return precision, recall, f1, auc_roc


# Random Predictor
def random_predictor(n_samples):
    return np.random.choice([0, 1], size=(n_samples,))


# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load Graph
version = "2000_nodes"
file = f"natural/{version}/final_graph_sequential_mean_euclidean_mixed_0.8_noisy.p"
g = pickle.load(open(file, "rb"))

# Convert node attributes to PyTorch tensor
for node, data in g.nodes(data=True):
    data['opinion'] = torch.tensor(data['opinion'])

# Convert the graph into PyG Data object
data_nd = from_networkx(g)
data = data_nd.to(device)
data.x = data.opinion

# Add self loops and convert to undirected graph
data.edge_index = to_undirected(data.edge_index).to(device)

# Create positive and negative links for training and testing
transform = RandomLinkSplit(is_undirected=False)
train_data, val_data, test_data = transform(data)


# Define Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GraphSAGE(data.num_features, 32).to(device)

"parameter to control whether training new model or loading existing model"

continue_training = "False"

if continue_training == "True":
    saved_model_path = "predictors_new/predictor_2000_nodes.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

    if os.path.isfile(saved_model_path):
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']


elif continue_training == "False":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.05)
    start_epoch = 0

epochs = 20
record_every = int(epochs / 20)


# Get link labels
def get_link_labels(edge_label_index, edge_label):
    return edge_label.to(device)


# Train Function
def train(data):
    # Code for training
    model.train()

    num_nodes = data.num_nodes
    num_neg_samples = data.edge_index.shape[1]  # Generate as many negative samples as there are positive edges

    # Generate negative edges
    neg_edge_index = utils.negative_sampling(edge_index=data.edge_index,
                                             num_nodes=num_nodes,
                                             num_neg_samples=num_neg_samples,
                                             method="sparse")

    # Concatenate positive and negative edges
    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)

    # Create corresponding labels: 1 for positive edges, 0 for negative edges
    edge_label = torch.cat([torch.ones(data.edge_index.shape[1]),
                            torch.zeros(num_neg_samples)], dim=0)

    # Add edge_label and edge_label_index to your data object
    data.edge_label = edge_label.to(device)
    data.edge_label_index = edge_label_index.to(device)

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
    predictions = (torch.sigmoid(link_logits) > 0.75).long()
    accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)

    # Additional metrics

    precision, recall, f1, auc_roc = get_metrics(link_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                                 torch.sigmoid(link_logits).cpu().detach().numpy())

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy, precision, recall, f1, auc_roc


# Test Function
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

    predictions = (torch.sigmoid(link_logits) > 0.75).long()
    accuracy = (predictions == link_labels.long()).sum().item() / link_labels.size(0)

    precision, recall, f1, auc_roc = get_metrics(link_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                                 torch.sigmoid(link_logits).cpu().detach().numpy())

    # including random predictions

    random_predictions = random_predictor(len(predictions))
    rand_precision, rand_recall, rand_f1, rand_auc_roc = get_metrics(link_labels.cpu().detach().numpy(),
                                                                     random_predictions,
                                                                     np.random.uniform(0, 1, len(predictions)))

    return loss.item(), accuracy, precision, recall, f1, auc_roc, rand_precision, rand_recall, rand_f1, rand_auc_roc


# Training Loop
epochs = 10
record_every = int(epochs / 10)

train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1s = []
train_aucs = []

val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []
val_aucs = []

test_losses = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1s = []
test_aucs = []



for epoch in tqdm(range(start_epoch, epochs + 1)):
# Training, Validation and Testing code goes here
    train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc = train(train_data)

    if epoch % record_every == 0:
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_aucs.append(train_auc)

        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, \
            rand_precision_val, rand_recall_val, rand_f1, rand_auc_roc= test(val_data)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)

        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc, \
            rand_precision, rand_recall, rand_f1, rand_auc_roc = test(test_data)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)

        print(" ")
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Precision: {train_precision:.4f}, '
              f'Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, '
              f'Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, '
              f'Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')
        print(f'Rand Precision: {rand_precision:.4f}, '
              f'Rand Recall: {rand_recall:.4f}, Rand F1: {rand_f1:.4f}, Rand AUC: {rand_auc_roc:.4f}')

    # Model Saving
    name = f"predictor_{version}.pth"
    torch.save(model.state_dict(), f"predictors_new/{name}_clean")

    saved_model_path = f"predictors_new/{name}"
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, saved_model_path)

# Plotting
def plot_metrics(metric_name):
    # Plot losses and accuracies
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(record_every, epochs + 1, record_every), train_losses, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_losses, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_losses, label='Test')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(record_every, epochs + 1, record_every), train_accuracies, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_accuracies, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_accuracies, label='Test')
    plt.title('Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot the rest of the metrics
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(record_every, epochs + 1, record_every), train_precisions, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_precisions, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_precisions, label='Test')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(record_every, epochs + 1, record_every), train_recalls, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_recalls, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_recalls, label='Test')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(record_every, epochs + 1, record_every), train_f1s, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_f1s, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_f1s, label='Test')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(record_every, epochs + 1, record_every), train_aucs, label='Training')
    plt.plot(range(record_every, epochs + 1, record_every), val_aucs, label='Validation')
    plt.plot(range(record_every, epochs + 1, record_every), test_aucs, label='Test')
    plt.title('AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

plot_metrics('losses')
plot_metrics('accuracies')
plot_metrics('precisions')
plot_metrics('recalls')
plot_metrics('f1s')
plot_metrics('aucs')
