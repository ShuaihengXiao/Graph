from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import itertools
import scipy.sparse as sp
import networkx as nx

"""
   Implementing GraphSAGE Algorithm Using Backend PyTorch
   @Author: ShuaihengXiao 
   @Nov10,2021
   
   torch==1.9.1
   scipy==1.5.4
   networkx==2.5.1
   
   Args:
   
   node_df -> pd.DataFrame
               Necessary Format:
               --------------------------------------------------------
               cust_id | is_driver | is_reported | feat_1 | feat_2 |...
               --------------------------------------------------------
   edge_df -> pd.DataFrame
               Necessary Format:
               ------------------
               cust_id | opp_id
               ------------------
   hidden_dim -> list(int)
               list of dimensions in each hidden layers
   
   num_neighbors_list -> list(int)
               list of number of neighbor nodes be sampled. Noted that length of hidden_dim must equaly to length of num_neightbors_list
               
   aggr_neighbor_method -> string
               method of neighbor aggregator, ["mean", "sum", "max"], default:mean           
    
   aggr_hidden_method -> string
               method of node embedding update, ["sum", "concat"], default: sum
   
   batch_size -> int
               batch size, default: 16
   
   epochs -> int
               default: 20
   
   num_batch_per_epoch -> int
               number of batch in each epoch default: 20
               
   lr -> float
               learning rate default: 0.01
         
   residual_block -> boolean
               model including residual block or not, default: True
               
   save_path -> string
               saving path, default: None
   
   seed -> int
               seed, default: 42
   
      
   
   v3 update:
   1. import networkx for adjacency_dict and adjacency_matrix building, compute more efficient
   
   v2 update:
   1. fix dimension dismatch when aggr_hidden_method = concat 
   2. fix when 'aggr_neighbor_method = max' cause wrong return type problem
   3. adding adjacency normalizaion \hat{A} =D^-0.5 * (A+I) * D^-0.5, normally in evey batch, Z = AXW, where A is normalized.
      for the sake of computing efficency we set AX = X in the initial step. now Z = XW
   
   v1 update:
   1. add some utalization module i.e. residual block, batch normalization 
"""
class utils(object):

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    @staticmethod
    def build_graph(node_df, edge_df):

        # build undirected weighted graph
        graph = nx.Graph()
        for index in edge_df.index:
            graph.add_edge(int(edge_df.loc[index]['cust_id']),
                           int(edge_df.loc[index]['opp_id']),
                           weight = edge_df.loc[index]['weight'])

        # build node-index lookup table 
        node_lookup = pd.DataFrame({'node': node_df.index}, index = node_df.cust_id)

        # delete isolates
        diff_node = list(graph.nodes())
        node_df = node_df.iloc[node_lookup.iloc[diff_node]['node']].reset_index(drop = True)

        # rebuild node-index lookup table 
        node_lookup = pd.DataFrame({'node': node_df.index}, index = node_df.cust_id)

        # adjacency index dict
        adj_dict_index = defaultdict(list)
        for n, dic in graph.adjacency():
            for dkey, _ in dic.items():
                adj_dict_index[node_lookup.loc[n]['node']].append(node_lookup.loc[dkey]['node'])

        # feature matrix 
        x = node_df[set(node_df) - {'cust_id', 'is_driver', 'is_reported'}].to_numpy()
        y = node_df.is_reported.to_numpy() * 1

        # mask config
        train_mask = node_df.is_driver.to_numpy()
        test_mask = ~train_mask

        return x, y, adj_dict_index, train_mask, test_mask, graph

    @staticmethod
    def get_adjacency_matrix(graph):

        return nx.adjacency_matrix(graph)

    @staticmethod
    def normalization(adjacency):
        """
            calculate \hat{A} = D^-0.5 * (A+I) * D^-0.5
        """
        adjacency += sp.eye(adjacency.shape[0])    
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        
        return d_hat.dot(adjacency).dot(d_hat).tocoo()
        
    @staticmethod
    def adjacency2tensor(x,adjacency):
        """
            convert numpy.array adjacency matrix to torch.tensor
        """
        num_nodes, input_dim = x.shape
        indices = torch.from_numpy(np.asarray([adjacency.row, 
                                               adjacency.col]).astype('int64')).long()
        values = torch.from_numpy(adjacency.data.astype(np.float32))
        tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                                    (num_nodes, num_nodes))
        return tensor_adjacency

    @staticmethod
    def sampling(src_nodes, sample_num, neighbor_table):
        """sampling neighbor nodes with fixed number according to src_node, note: this is an overload sample
        if sum(num of neighbor nodes) less than fixed number, would replace with same nodes
        
        paper:  ''In this work, we uniformly sample a fixed-size set of neighbors, instead of
                using full neighborhood sets in Algorithm 1, in order to keep the computational footprint of each batch fixed. 
                That is, using overloaded notation'' -- <Inductive Representation Learning on Large Graphs>
        
        Arguments:
            src_nodes {list, ndarray} -- src nodes list
            sample_num {int} -- num of nerghbor nodes needa sampled
            neighbor_table {dict} -- adjacency_dict
        
        Returns:
            np.ndarray 
        """
        results = []
        for sid in src_nodes:
            # overload neighbor sampling
            res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
            results.append(res)
        return np.asarray(results).flatten()


    @staticmethod
    def multihop_sampling(src_nodes, sample_nums, neighbor_table):
        """multihop_sampling -- layer-wise sampling
        
        Arguments:
            src_nodes {list, np.ndarray} -- src nodes list
            sample_nums {list of int} -- num of nerghbor nodes needa sampled in the hop
            neighbor_table {dict} -- adjacency_dict
        
        Returns:
            [list of ndarray]
        """
        sampling_result = [src_nodes]
        for k, hopk_num in enumerate(sample_nums):
            hopk_result = utils.sampling(sampling_result[k], hopk_num, neighbor_table)
            sampling_result.append(hopk_result)
        return sampling_result



class NeighborAggregator(nn.Module):
    def __init__ (self, 
                  input_dim, 
                  output_dim, 
                  aggr_method,
                  aggr_hidden_method,
                  use_bias=False):
        """NeighborAggregator:

        Args:
            input_dim: input dim
            output_dim: output dim 
            use_bias: import bias or not (default: {False})
            aggr_method: method of neighbor aggregator (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.aggr_hidden_method = aggr_hidden_method
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)[0]
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method)) 
        
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
    

class SageGCN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 residual_block,
                 aggr_neighbor_method,
                 aggr_hidden_method,
                 activation = F.relu):
        
        """SageGCN statement:

        Args:
            input_dim: input dimension
            hidden_dim: hidden layer dimension
                if aggr_hidden_method = sum, output dimension is hidden_dim
                if aggr_hidden_method = concat, output dimension is hidden_dim * 2
            activation: default: relu
            aggr_neighbor_method: method of neighbor aggregator, ["mean", "sum", "max"], default:mean
            aggr_hidden_method: method of node embedding update, ["sum", "concat"], default: sum
        """
        super(SageGCN, self).__init__()
       
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        
        # double hidden_dim if concat
        if self.aggr_hidden_method == "concat":
            self.hidden_dim = self.hidden_dim * 2
        
        self.activation = activation
        self.residual_block = residual_block
        
        self.aggregator = NeighborAggregator(self.input_dim, self.hidden_dim,
                                             aggr_method = self.aggr_neighbor_method,
                                            aggr_hidden_method = self.aggr_hidden_method)
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, hidden_dim))
        self.reset_parameters()
         
        
        if self.residual_block:
            
            if self.aggr_hidden_method == "concat":
                self.linear_1 = nn.Linear(self.hidden_dim // 2 * 3,self.hidden_dim // 2)
                self.linear_2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                self.bn_1 = nn.BatchNorm1d(self.hidden_dim // 2)
                self.bn_2 = nn.BatchNorm1d(self.hidden_dim // 2)
            elif self.aggr_hidden_method == "sum":
                self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
                self.linear_2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                self.bn_1 = nn.BatchNorm1d(self.hidden_dim * 2)
                self.bn_2 = nn.BatchNorm1d(self.hidden_dim)

            torch.nn.init.xavier_uniform_(self.linear_1.weight,
                                          gain = torch.nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.linear_2.weight,
                                          gain = torch.nn.init.calculate_gain('relu'))
            
            
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):

        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
            
        ## add residual block
        if self.residual_block:
            if self.aggr_hidden_method == "concat":
                hidden = F.relu(self.linear_1(hidden))
                dummy_hidden = hidden
            elif self.aggr_hidden_method == "sum":
                dummy_hidden = hidden
                hidden = F.relu(self.linear_1(hidden))
            hidden = self.bn_1(hidden)
            hidden = F.relu(self.linear_2(hidden))
            hidden = self.bn_2(hidden)
            hidden = torch.add(hidden, dummy_hidden)

        
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden
        

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    def __init__(self, input_dim, 
                       hidden_dim,
                       num_neighbors_list, 
                       aggr_neighbor_method, 
                       aggr_hidden_method,
                       residual_block):
        super(GraphSage, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.num_layers = len(num_neighbors_list)
        self.residual_block = residual_block
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, 
                                hidden_dim[0],
                                residual_block = self.residual_block,
                                aggr_neighbor_method = self.aggr_neighbor_method,
                                aggr_hidden_method = self.aggr_hidden_method))
        
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], 
                                    hidden_dim[index+1],
                                    residual_block = self.residual_block,
                                    aggr_neighbor_method = self.aggr_neighbor_method,
                                    aggr_hidden_method = self.aggr_hidden_method))
        self.gcn.append(SageGCN(hidden_dim[-2], 
                                hidden_dim[-1], 
                                residual_block = self.residual_block, 
                                aggr_neighbor_method = self.aggr_neighbor_method,
                                aggr_hidden_method = self.aggr_hidden_method,
                                activation = None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list)

class run_model():
    
    def __init__(self,
                 node_df,
                 edge_df,
                 hidden_dim = [128, 2],
                 num_neighbors_list = [10, 10],
                 aggr_neighbor_method = 'mean',
                 aggr_hidden_method = "sum",
                 batch_size = 16,
                 epochs = 20,
                 num_batch_per_epoch = 20,
                 lr = 0.01,
                 residual_block = True,
                 save_path = None,
                 seed = 42):
        
        assert isinstance(node_df, pd.DataFrame)
        assert isinstance(edge_df, pd.DataFrame)
        assert isinstance(hidden_dim, list)
        assert isinstance(num_neighbors_list, list)
        assert len(hidden_dim) == len(num_neighbors_list)
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        
        self.node_df = node_df
        self.edge_df = edge_df
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_batch_per_epoch = num_batch_per_epoch
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.residual_block = residual_block
        self.save_path = save_path
        self.seed = seed
        
        # set seed
        utils.set_seed(self.seed)
        
        # build graph, feature matrix, adjacency dict and train test mask
        print('data preprocessing..\n')
        self.x,self.y,self.adjacency_dict,self.train_mask,self.test_mask, self.graph = utils.build_graph(self.node_df,self.edge_df)
        
        # build normalized adjacency matrix 
        self.adjacency = utils.get_adjacency_matrix(self.graph)
        self.adjacency = utils.normalization(self.adjacency)
        self.adjacency = utils.adjacency2tensor(self.x, self.adjacency)
        self.x = torch.from_numpy(self.x).float().to(self.device)
        self.x = torch.sparse.mm(self.adjacency,self.x)
        
        print('after filtering single nodes \n')
        print('num of train instances: {}'.format(sum(self.train_mask)))
        print('num of test instances: {}'.format(sum(self.test_mask)))
        
        # build model 
        self.model = GraphSage(input_dim = self.x.shape[1],
                          hidden_dim = self.hidden_dim,
                          num_neighbors_list = self.num_neighbors_list,
                          aggr_neighbor_method = self.aggr_neighbor_method,
                          aggr_hidden_method = self.aggr_hidden_method,
                          residual_block = self.residual_block).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        print('model structure')
        print(self.model)
        
    
    def train(self):
        
        train_index = np.where(self.train_mask)[0]
        train_label = self.y
        
        print('model training..')
        print('training through {}'.format(self.device))
        self.model.train()
        for e in range(self.epochs):
            for batch in range(self.num_batch_per_epoch):
                batch_src_index = np.random.choice(train_index, size=(self.batch_size,))
                batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(self.device)
                batch_sampling_result = utils.multihop_sampling(batch_src_index, self.num_neighbors_list, self.adjacency_dict)
                #print('batch_sampling_result',batch_sampling_result)
                #batch_sampling_x = [torch.from_numpy(self.x[idx]).float().to(self.device) for idx in batch_sampling_result]
                batch_sampling_x = [self.x[idx] for idx in batch_sampling_result]
                batch_train_logits = self.model(batch_sampling_x).squeeze(1)
                loss = self.criterion(batch_train_logits, batch_src_label)
                self.optimizer.zero_grad()
                loss.backward()  
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 5.0)
                self.optimizer.step()  
                print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
            self.test()
        print('training complete!')
        
        if self.save_path:
            torch.save(self.model,self.save_path)
            print('model saved!')
        
    def test(self):
        test_index = np.where(self.test_mask)[0]
        self.model.eval()
        with torch.no_grad():
            test_sampling_result = utils.multihop_sampling(test_index, self.num_neighbors_list, self.adjacency_dict)
            test_x = [self.x[idx] for idx in test_sampling_result]
            test_logits = self.model(test_x).squeeze(1)
            test_label = torch.from_numpy(self.y[test_index]).long().to(self.device)
            predict_y = test_logits.max(1)[1]
            #predict_y = (test_logits > 0.5).float()
            accuarcy = torch.eq(predict_y, test_label).float().mean().item()
            
            print("test accuracy: ", accuarcy)
            
    def __version__(self):
        return 'v3'
