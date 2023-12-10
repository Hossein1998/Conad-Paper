import torch
import networkx as nx
from scipy.sparse import data
import torch
import torch.nn.functional as F
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import inv, fractional_matrix_power
import dgl
import numpy as np
import os
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import community as community_louvain
red = "\033[0;31m"



def load_anomaly_detection_dataset(dataset):
    
    data_mat = scipy.io.loadmat(f'{dataset}/{dataset}.mat')
    adj = data_mat.get('A', data_mat.get('Network'))
    feat = data_mat.get('X', data_mat.get('Attributes'))
    truth = data_mat.get('gnd', data_mat.get('Label')).flatten()
    
    # Convert to dense format if they are sparse matrices
    adj = adj.toarray() if not isinstance(adj, np.ndarray) else adj
    feat = feat.toarray() if not isinstance(feat, np.ndarray) else feat
    
    return adj, feat, truth



def make_anomalies(adj, feat, rate=.3, sourround=50, scale_factor=10, noise_level=0.2):
    # Convert feat_aug to float64
    adj_aug, feat_aug = adj.copy(), feat.copy().astype('float64')
    label_aug = np.zeros(adj.shape[0])
    assert(adj_aug.shape[0] == feat_aug.shape[0])
    num_nodes = adj_aug.shape[0]

    # Calculate average degree of the network
    avg_degree = np.mean(np.sum(adj, axis=1))

    # Convert adj to graph and perform community detection
    G = nx.from_numpy_array(adj)
    communities = community_louvain.best_partition(G)

    for i in range(num_nodes):
        prob = np.random.uniform()
        if prob > rate:
            continue
        label_aug[i] = 1

        # Randomly choose a type of anomaly
        anomaly_type = np.random.randint(0, 8)  # Adjusted to include new anomaly type

        # anomaly_type = 8

        if anomaly_type == 0:
            # Add clique
            min_clique_size = max(3, int(avg_degree * 0.5))
            max_clique_size = max(min_clique_size + 1, int(avg_degree * 1.5))
            clique_size = np.random.randint(min_clique_size, max_clique_size)
            less_connected_nodes = np.where(np.sum(adj, axis=1) < avg_degree)[0]
            if len(less_connected_nodes) > 0:
                node_for_clique = np.random.choice(less_connected_nodes)
                new_neighbors = np.random.choice(np.arange(num_nodes), clique_size, replace=False)
                for n in new_neighbors:
                    if n != node_for_clique:
                        adj_aug[n][node_for_clique] = 1
                        adj_aug[node_for_clique][n] = 1

        elif anomaly_type == 1:
            # Selective and Partial Edge Dropping
            neighbors = np.nonzero(adj[i])[0]
            if neighbors.size > 0:
                proportion_to_drop = 0.5
                num_edges_to_drop = int(proportion_to_drop * len(neighbors))
                neighbor_degrees = np.sum(adj[neighbors, :], axis=1)
                neighbors_sorted_by_degree = neighbors[np.argsort(-neighbor_degrees)]
                edges_to_drop = neighbors_sorted_by_degree[:num_edges_to_drop]
                for n in edges_to_drop:
                    adj_aug[i][n] = 0
                    adj_aug[n][i] = 0

        elif anomaly_type == 2:
            # Contextual Attribute Modification and Combining Attribute Changes
            non_neighbors_mask = np.sum(adj[i] + adj[:, i], axis=0) == 0
            non_neighbors = np.atleast_1d(non_neighbors_mask).nonzero()[0]
            direct_neighbors_mask = adj[i].astype(bool)
            direct_neighbors = np.atleast_1d(direct_neighbors_mask).nonzero()[0]
            candidates = np.concatenate([non_neighbors, direct_neighbors]) if len(direct_neighbors) > 0 else non_neighbors
            if len(candidates) > 0:
                chosen_node = np.random.choice(candidates)
                num_attrs = feat_aug.shape[1]
                selected_attrs = np.random.choice(num_attrs, num_attrs // 2, replace=False)
                feat_aug[i, selected_attrs] = feat[chosen_node, selected_attrs]

        elif anomaly_type == 3:
            # Selective Scaling and Non-Uniform Scaling of Attributes
            num_attrs = feat_aug.shape[1]
            attr_variance = np.var(feat_aug, axis=0)
            important_attrs = np.argsort(-attr_variance)[:num_attrs // 2]
            for attr in important_attrs:
                scale_factor_attr = np.random.uniform(0.5, 1.5)
                feat_aug[i, attr] *= scale_factor_attr
            remaining_attrs = [attr for attr in range(num_attrs) if attr not in important_attrs]
            for attr in remaining_attrs:
                feat_aug[i, attr] *= scale_factor

        elif anomaly_type == 4:
            # Random Attribute Noise
            noise_attrs = np.random.choice(feat_aug.shape[1], feat_aug.shape[1] // 2, replace=False)
            noise = np.random.normal(0, noise_level, len(noise_attrs))
            feat_aug[i, noise_attrs] += noise

        elif anomaly_type == 5:
            # Structural Anomaly - Breaking Community Structures
            other_nodes = [node for node, comm in communities.items() if comm != communities[i]]
            if other_nodes:
                other_node = np.random.choice(other_nodes)
                adj_aug[i, other_node] = 1
                adj_aug[other_node, i] = 1

        elif anomaly_type == 6:
            # Structural Anomaly - Connecting Distant Nodes
            non_neighbors_mask = np.sum(adj[i] + adj[:, i], axis=0) == 0
            non_neighbors = np.atleast_1d(non_neighbors_mask).nonzero()[0]
            if len(non_neighbors) > 0:
                distant_node = np.random.choice(non_neighbors)
                adj_aug[i, distant_node] = 1
                adj_aug[distant_node, i] = 1

        elif anomaly_type == 7:
            # Random Edge Removal
            non_neighbors_mask = np.sum(adj[i] + adj[:, i], axis=0) == 0
            non_neighbors = np.atleast_1d(non_neighbors_mask).nonzero()[0]
            if len(non_neighbors) > 0:
                 distant_node = np.random.choice(non_neighbors)
                 adj_aug[i, distant_node] = 1
                 adj_aug[distant_node, i] = 1
                
        elif anomaly_type == 8:
            # Creating Isolated Subgraphs (Isolated Cliques)
            clique_size = np.random.randint(3, 6)  # Choose the clique size
            clique_nodes = np.random.choice(num_nodes, clique_size, replace=False)
            for node in clique_nodes:
                adj_aug[node, :] = 0
                adj_aug[:, node] = 0
                
            for node1 in clique_nodes:
                for node2 in clique_nodes:
                    if node1 != node2:
                         adj_aug[node1, node2] = 1

     
    return adj_aug, feat_aug, label_aug, anomaly_type




import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
import dgl.function as fn
import dgl.ops as ops
import scipy.sparse


class Reconstruct(nn.Module):
    '''reconstruct the adjacent matrix and rank anomalies'''
    def __init__(self, **kwargs):
        super(Reconstruct, self).__init__(**kwargs)

    def forward(self, h):
        return torch.mm(h, h.transpose(1, 0))

class GRL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_num=1, head_num=1) -> None:
        super(GRL, self).__init__()
        self.shared_encoder = nn.ModuleList(
            GraphConv(
                in_dim if i==0 else hidden_dim,
                (out_dim if i == hidden_num - 1 else hidden_dim),
                activation=torch.sigmoid
            )
            for i in range(hidden_num)
        )
        self.attr_decoder = GraphConv(
            in_feats=out_dim,
            out_feats=in_dim
            # activation=torch.sigmoid,
        )
        self.struct_decoder = nn.Sequential(
            Reconstruct(),
            nn.Sigmoid()
        )
        self.dense = nn.Sequential(nn.Linear(out_dim, out_dim))

    def embed(self, g, h):
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # h = self.project(g, h).view(h.shape[0], -1)
        # return h.div(torch.norm(h, p=2, dim=1, keepdim=True))
        return self.dense(h)
    
    def reconstruct(self, g, h):
        struct_reconstructed = self.struct_decoder(h)
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        return struct_reconstructed, x_hat

    def forward(self, g, h):
        # encode
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        # return reconstructed matrices
        return struct_reconstructed, x_hat



def loss_func(a, a_hat, x, x_hat, weight1=1, weight2=1, alpha=0.6, mask=1):
    # adjacency matrix reconstruction
    struct_weight = weight1
    struct_error_total = torch.sum(torch.square(torch.sub(a, a_hat)), axis=-1)
    struct_error_sqrt = torch.sqrt(struct_error_total) * mask
    struct_error_mean = torch.mean(struct_error_sqrt)
    # feature matrix reconstruction
    feat_weight = weight2
    feat_error_total = torch.sum(torch.square(torch.sub(x, x_hat)), axis=-1)
    feat_error_sqrt = torch.sqrt(feat_error_total) * mask
    feat_error_mean = torch.mean(feat_error_sqrt)
    loss =  (1 - alpha) * struct_error_sqrt + alpha * feat_error_sqrt
    # loss = struct_error_sqrt
    return loss, struct_error_mean, feat_error_mean


dataset = "Flickr"  
cuda = True
epoch1 = 200
lr = 1e-3
margin = 0.5


# input attributed network G
adj, attrs, label = load_anomaly_detection_dataset(dataset)
 
print(adj.shape)

# create graph and attribute object, as anchor point
graph1 = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).add_self_loop()
attrs1 = torch.FloatTensor(attrs)
num_attr = attrs.shape[1]
# hidden dimension, output dimension
hidden_dim, out_dim = 128, 64
hidden_num = 2

model = GRL(num_attr, hidden_dim, out_dim, hidden_num)



def criterion(z, z_hat, y, margin):
    # Vectorized computation of squared differences
    diff_squared = (z - z_hat) ** 2

    # Sum over the feature dimension to get a 1D tensor
    diff_squared = diff_squared.sum(dim=1)

    # Ensure y is a 1D tensor
    y = y.squeeze()

    # Create masks for different conditions
    anomaly_mask = y == 1
    normal_mask = ~anomaly_mask

    # Apply conditions using masks
    loss_normal = diff_squared[normal_mask]
    loss_anomaly = F.softplus(margin - diff_squared[anomaly_mask])

    # Combine losses
    total_loss = torch.cat((loss_normal, loss_anomaly)).mean()

    return total_loss




cuda_device = torch.device('cuda') if cuda else torch.device('cpu')
cpu_device = torch.device('cpu')
model = model.to(cuda_device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
sw = SummaryWriter('logs/siamese_%s_%s' % (dataset, t))
model.train()


adj_aug, attrs_aug, label_aug,i = make_anomalies(adj, attrs, sourround=50)

print(i)

graph2 = dgl.from_scipy(scipy.sparse.coo_matrix(adj_aug)).add_self_loop()
attrs2 = torch.FloatTensor(attrs_aug)

# train encoder with supervised contrastive learning
for i in range(epoch1):
    # augmented labels introduced by injection
    labels = torch.FloatTensor(label_aug).unsqueeze(-1)
    if cuda:
        graph1 = graph1.to(cuda_device)
        attrs1 = attrs1.to(cuda_device)
        graph2 = graph2.to(cuda_device)
        attrs2 = attrs2.to(cuda_device)
        labels = labels.to(cuda_device)

    # train siamese loss
    orig = model.embed(graph1, attrs1)
    aug = model.embed(graph2, attrs2)
    
    margin_loss = criterion(orig,aug,labels,margin)
    
    
    margin_loss = margin_loss.mean()
    sw.add_scalar('train/margin_loss', margin_loss, i)
    
    margin_loss_number = margin_loss.detach().cpu().item()
    if i % 20 == 0:
        print(f'Epoch {i}: Margin Loss = {margin_loss_number}')
    
    optimizer.zero_grad()
    margin_loss.backward()
    optimizer.step()

    # train reconstruction
    A_hat, X_hat = model(graph2, attrs2)
    a = graph1.adjacency_matrix().to_dense()
    recon_loss, struct_loss, feat_loss = loss_func(a.cuda() if cuda else a, A_hat, attrs1, X_hat, weight1=1, weight2=1, alpha=.7, mask=1)
    recon_loss = recon_loss.mean()
    # loss = bce_loss + recon_loss
    optimizer.zero_grad()
    recon_loss.backward()
    optimizer.step()
    sw.add_scalar('train/rec_loss', recon_loss, i)
    sw.add_scalar('train/struct_loss', struct_loss, i)
    sw.add_scalar('train/feat_loss', feat_loss, i)
    
    recon_loss_number = recon_loss.detach().cpu().item()
    if i % 20 == 0:
        print(f'\033[91mEpoch {i}: Margin Loss for reconstruction = {recon_loss_number}\033[0m')   
    
#print(_list[0])

# evaluate
with torch.no_grad():
    A_hat, X_hat = model(graph2, attrs2)
    A_hat, X_hat = A_hat.cpu(), X_hat.cpu()
    a = graph1.adjacency_matrix().to_dense().cpu()
    recon_loss, struct_loss, feat_loss = loss_func(a, A_hat, attrs1.cpu(), X_hat, weight1=1, weight2=1, alpha=.7)
    score = recon_loss.detach().numpy()
    # score = (score - score.min()) / (score.max() - score.min())
    print('AUC: %.4f' % roc_auc_score(label, score))
    # for k in [50, 100, 200, 300]:
    #     print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))