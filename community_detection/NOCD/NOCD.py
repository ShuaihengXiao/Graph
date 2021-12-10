import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch.nn.functional as F

'''
    implementing NOCD algorithm with backend pytorch
    
    args:
        1. node_df -> pandas.dataframe storing nodes feature matrix 
        2. edge_df -> pandas.dataframe storing edge information
        3. hidden_sizes -> list hidden size of the GCN, default: [128]
        4. num_communities -> int number of communities, default: 20
        5. weight_decay -> float strength of L2 regularization on GCN weights, default: 1e-2
        6. dropout -> float dropout rate, default: 0.5
        7. batch_norm -> bool whether to use batch norm, default: True
        8. lr -> float learning rate, default: 1e-3
        9. max_epochs -> int epoch in training step, default: 500
        10. balance_loss -> bool whether to use balanced loss, default: True
        11. stochastic_loss -> bool whether to use stochastic or full-batch training, default: True
        12. batch_size -> int batch size (only for stochastic training), default: 20000
'''


class preprocessing(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_adj(edge_df):
        
        graph = nx.from_edgelist([(cust,opp) for cust, opp in zip(edge_df['cust_id'],edge_df['opp_id'])])
        
        return nx.adjacency_matrix(graph),graph.number_of_nodes()
    
    @staticmethod
    def del_nodes(node_df,edge_df):
        # node_lookup: store node index
        node_lookup = pd.DataFrame({'node': node_df.index,}, index=node_df.cust_id)

        # delete no-edge-node 
        diff_node = list(set(node_df['cust_id'])-(set(node_df['cust_id']) - set(edge_df['cust_id']) - set(edge_df['opp_id'])))

        node_df = node_df.iloc[node_lookup.iloc[diff_node]['node']].reset_index(drop=True)
        return node_df
    
    @staticmethod
    def sklearn_normalize(matrix):
        
        return normalize(matrix)
    
    @staticmethod
    def to_sparse_tensor(matrix, cuda: bool = False,):
        """Convert a scipy sparse matrix to a torch sparse tensor.

        Args:
            matrix: Sparse matrix to convert.
            cuda: Whether to move the resulting tensor to GPU.

        Returns:
            sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

        """
        if sp.issparse(matrix):
            coo = matrix.tocoo()
            indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
            values = torch.FloatTensor(coo.data)
            shape = torch.Size(coo.shape)
            sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        elif torch.is_tensor(matrix):
            row, col = matrix.nonzero().t()
            indices = torch.stack([row, col])
            values = matrix[row, col]
            shape = torch.Size(matrix.shape)
            sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        else:
            raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
        if cuda:
            sparse_tensor = sparse_tensor.cuda()
        return sparse_tensor.coalesce()
    
    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return preprocessing.to_sparse_tensor(adj_norm)
    
class EdgeSampler(torch.utils.data.Dataset):
    """Sample edges and non-edges uniformly from a graph.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
    """
    def __init__(self, A, num_pos=1000, num_neg=1000):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A
        self.edges = np.transpose(A.nonzero())
        self.num_nodes = A.shape[0]
        self.num_edges = self.edges.shape[0]

    def __getitem__(self, key):
        np.random.seed(key)
        edges_idx = np.random.randint(0, self.num_edges, size=self.num_pos, dtype=np.int64)
        next_edges = self.edges[edges_idx, :]

        # Select num_neg non-edges
        generated = False
        while not generated:
            candidate_ne = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
#             to_keep = (1 - self.A[cne1, cne2]).astype(np.bool).A1 * (cne1 != cne2)
            to_keep = np.multiply((1 - self.A[cne1, cne2]).astype(np.bool),np.matrix((cne1 != cne2).astype(np.bool)))
            to_keep = np.ravel(to_keep)
            next_nonedges = candidate_ne[to_keep][:self.num_neg]
            generated = to_keep.sum() >= self.num_neg
        return torch.LongTensor(next_edges), torch.LongTensor(next_nonedges)

    def __len__(self):
        return 2**32
    
    @staticmethod
    def collate_fn(batch):
        edges, nonedges = batch[0]
        return (edges, nonedges)
    
    @staticmethod
    def get_edge_sampler(A, num_pos=1000, num_neg=1000):
        data_source = EdgeSampler(A, num_pos, num_neg)
        return torch.utils.data.DataLoader(data_source, collate_fn = EdgeSampler.collate_fn)

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
#         return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

class BerpoDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super(BerpoDecoder).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        """Compute BerPo loss for all edges & non-edges in a graph."""
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)
    
    @staticmethod
    def l2_reg_loss(model, scale=1e-5):
        """Get L2 loss for model weights."""
        loss = 0.0
        for w in model.get_weights():
            loss += w.pow(2.).sum()
        return loss * scale

class run_model(object):
    
    def __init__(self,
                node_df,
                edge_df,
                hidden_sizes = [128],
                num_communities = 20,
                weight_decay = 1e-2,
                dropout = 0.5,
                batch_norm = True,
                lr = 1e-3,
                max_epochs = 500,
                balance_loss = True,
                stochastic_loss = True,
                batch_size = 20000):
        
        self.node_df = node_df
        self.edge_df = edge_df
        self.hidden_sizes = hidden_sizes
        self.num_communities = num_communities
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.max_epochs = max_epochs
        self.balance_loss = balance_loss
        self.stochastic_loss = stochastic_loss
        self.batch_size = batch_size
        
    
        print('preprocessing step')
        self.A,self.N = preprocessing.get_adj(self.edge_df)
        self.node_df = preprocessing.del_nodes(self.node_df,self.edge_df)
        self.x_norm = preprocessing.sklearn_normalize(self.node_df)
        self.x_norm = preprocessing.to_sparse_tensor(sp.csr_matrix(self.x_norm))
        self.sampler = EdgeSampler.get_edge_sampler(self.A, self.batch_size, self.batch_size)
        self.gnn = GCN(self.x_norm.shape[1], 
                  self.hidden_sizes, 
                  self.num_communities, 
                  batch_norm = self.batch_norm, 
                  dropout = self.dropout)
        self.adj_norm = preprocessing.normalize_adj(self.A)
        self.decoder = BerpoDecoder(self.N, self.A.nnz, balance_loss = self.balance_loss)
        self.opt = torch.optim.Adam(self.gnn.parameters(), lr = self.lr)
        
    def train(self):
        for epoch, batch in enumerate(self.sampler):

            if epoch > self.max_epochs:
                break
            if epoch % 25 == 0:

                with torch.no_grad():
                    self.gnn.eval()

                    self.Z = F.relu(self.gnn(self.x_norm, self.adj_norm))
                    val_loss = self.decoder.loss_full(self.Z, self.A)

                    print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}')

            # Training step
            self.gnn.train()

            self.opt.zero_grad()

            self.Z = F.relu(self.gnn(self.x_norm, self.adj_norm))

            ones_idx, zeros_idx = batch
            if self.stochastic_loss:
                loss = self.decoder.loss_batch(self.Z, ones_idx, zeros_idx)
                loss = self.decoder.loss_full(self.Z, self.A)
            loss += BerpoDecoder.l2_reg_loss(self.gnn, scale = self.weight_decay)

            loss.backward()
            self.opt.step()