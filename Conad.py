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
import torch.nn.functional as F
import dgl
import os
import numpy as np
import networkx as nx
from scipy.sparse import data
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import inv, fractional_matrix_power




def load_anomaly_detection_dataset(dataset):
    
    data_mat = scipy.io.loadmat(f'{dataset}/{dataset}.mat')
    adj = data_mat.get('A', data_mat.get('Network'))
    feat = data_mat.get('X', data_mat.get('Attributes'))
    truth = data_mat.get('gnd', data_mat.get('Label')).flatten()
    
    # Convert to dense format if they are sparse matrices
    adj = adj.toarray() if not isinstance(adj, np.ndarray) else adj
    feat = feat.toarray() if not isinstance(feat, np.ndarray) else feat
    
    return adj, feat, truth


def make_anomalies(adj, feat, rate=.1, clique_size=30, sourround=50, scale_factor=10):
    # Convert feat_aug to float64
    adj_aug, feat_aug = adj.copy(), feat.copy().astype('float64')
    label_aug = np.zeros(adj.shape[0])
    assert(adj_aug.shape[0]==feat_aug.shape[0])
    num_nodes = adj_aug.shape[0]
    for i in range(num_nodes):
        prob = np.random.uniform()
        if prob > rate: continue
        label_aug[i] = 1
        one_fourth = np.random.randint(0, 4)
        if one_fourth == 0:
            # add clique
            degree = np.sum(adj[i])
            max_clique_size=30
            new_neighbors = np.random.choice(np.arange(num_nodes), clique_size, replace=False)
            for n in new_neighbors:
                adj_aug[n][i] = 1
                adj_aug[i][n] = 1
        elif one_fourth == 1:
            # drop all connection
            neighbors = np.nonzero(adj[i])[0]
            if neighbors.size == 0:
                    continue
            elif neighbors.size == 1:
                    neighbors = [neighbors.item()]
            for n in neighbors:
                adj_aug[i][n] = 0
                adj_aug[n][i] = 0
        elif one_fourth == 2:
            # attrs
            candidates = np.random.choice(np.arange(num_nodes), sourround, replace=False)
            max_dev, max_idx = 0, i
            for c in candidates:
                dev = np.square(feat[i]-feat[c]).sum()
                if dev > max_dev:
                    max_dev = dev
                    max_idx = c
            feat_aug[i] = feat[max_idx]
        else:
            # scale attr
            prob = np.random.uniform(0, 1)
            if prob > 0.5:
                feat_aug[i] *= scale_factor
            else:
                feat_aug[i] /= scale_factor
    return adj_aug, feat_aug, label_aug



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
            out_feats=in_dim,
            #activation=torch.sigmoid,
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




dataset = "Amazon"  
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
    n = len(z)
    total_loss = 0.0
    _list = []

    for i in range(n):
        diff_squared = (z[i] - z_hat[i]) ** 2
        _list.append(diff_squared)

        if y[i] == 0:
            loss = diff_squared
        else:  # y[i] == 1
            # Soft margin modification
            loss = F.softplus(margin - diff_squared)

        total_loss += loss

    return total_loss / n, _list

cuda_device = torch.device('cuda') if cuda else torch.device('cpu')
cpu_device = torch.device('cpu')
model = model.to(cuda_device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
sw = SummaryWriter('logs/siamese_%s_%s' % (dataset, t))
model.train()


adj_aug, attrs_aug, label_aug = make_anomalies(adj, attrs, clique_size=20, sourround=50)

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
    margin_loss, _list = criterion(orig, aug, labels,margin)
        
    margin_loss = margin_loss.mean()
    sw.add_scalar('train/margin_loss', margin_loss, i)
    
    margin_loss_number = margin_loss.detach().cpu().item()
    if i % 5 == 0:
        print(f'Epoch {i}: Margin Loss = {margin_loss_number}')
    
    optimizer.zero_grad()
    margin_loss.backward()
    optimizer.step()

    # train reconstruction
    A_hat, X_hat = model(graph1, attrs1)
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
    if i % 5 == 0:
        print(f'Epoch {i}: Margin Loss for reconstruction = {recon_loss_number}')
    
    
#print(_list[0])

# evaluate
with torch.no_grad():
    A_hat, X_hat = model(graph2, attrs2)
    A_hat, X_hat = A_hat.cpu(), X_hat.cpu()
    a = graph1.adjacency_matrix().to_dense().cpu()
    recon_loss, struct_loss, feat_loss = loss_func(a, A_hat, attrs1.cpu(), X_hat, weight1=1, weight2=1, alpha=.7)
    score = recon_loss.detach().numpy()
    print('AUC: %.4f' % roc_auc_score(label, score))
    # for k in [50, 100, 200, 300]:
    #     print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))