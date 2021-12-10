from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

"""
   applying graphSAGE using backbend Pytorch
   author: Shuaiheng Xiao
   
   
   to do list:
   1. batch norm
   2. model save
"""

def data_format_process(node_df,edge_df):
    """
    output
    x: node features 
    y: label 
    adjacency_dict: dict which store neighbor's info.  {node_index:[neighbor1,neighbor2..]}
    
    """
    # node_lookup: store node index
    node_lookup = pd.DataFrame({'node': node_df.index,}, index=node_df.cust_id)
    
    # delete no-edge-node 
    diff_node = list(set(node_df['cust_id'])-(set(node_df['cust_id']) - set(edge_df['cust_id']) - set(edge_df['opp_id'])))
    
    node_df = node_df.iloc[node_lookup.iloc[diff_node]['node']].reset_index(drop=True)
    
    # build neighbor dictionary
    node_lookup = pd.DataFrame({'node': node_df.index,}, index=node_df.cust_id)
    adjacency_dict = defaultdict(list)
    for cust,opp in zip(edge_df['cust_id'],edge_df['opp_id']):
        adjacency_dict[node_lookup.loc[cust]['node']].append(node_lookup.loc[opp]['node'])
    
    # to array
    x = node_df[set(node_df) - {'cust_id', 'is_driver', 'is_reported'}].to_numpy()
    y = node_df.is_reported.to_numpy() * 1
    
    # building mask
    train_mask = node_df.is_driver.to_numpy()
    test_mask = ~train_mask
    
    return x,y,adjacency_dict,train_mask,test_mask


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


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """multihop_sampling
    
    Arguments:
        src_nodes {list, np.ndarray} -- src nodes list
        sample_nums {list of int} -- num of nerghbor nodes needa sampled in the hop
        neighbor_table {dict} -- adjacency_dict
    
    Returns:
        [list of ndarray]
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 use_bias=False, aggr_method="mean"):
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
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden, aggr_neighbor

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
    

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation = F.relu,
                 aggr_neighbor_method = "mean",
                 aggr_hidden_method = "sum"):
        """SageGCN statement:

        Args:
            input_dim: input dimension
            hidden_dim: hidden layer dimension
                if aggr_hidden_method = sum, output dimension is hidden_dim
                if aggr_hidden_method = concat, output dimension is hidden_dim * 2
            activation: defult = relu
            aggr_neighbor_method: method of neighbor aggregator, ["mean", "sum", "max"], default:mean
            aggr_hidden_method: method of node embedding update, ["sum", "concat"], default: sum
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden,aggr_neighbor = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

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
                 node_df,edge_df,
                 hidden_dim = [128, 2],
                 num_neighbors_list = [10, 10],
                 batch_size = 16,
                 epochs = 20,
                 num_batch_per_epoch = 20,
                 lr = 0.01):
        assert len(hidden_dim) == len(num_neighbors_list)
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_batch_per_epoch = num_batch_per_epoch
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print('data preprocessing..')
        self.x,self.y,self.adjacency_dict,self.train_mask,self.test_mask = data_format_process(node_df,edge_df)
        print('data preprocessing complete!')
        print('-----------------------*-----------------------')
        print('train dataset: {}'.format(sum(self.train_mask)))
        print('test dataset: {}'.format(sum(self.test_mask)))
        
        print('-----------------------*-----------------------')
        
        self.model = GraphSage(input_dim = self.x.shape[1],
                          hidden_dim = self.hidden_dim,
                          num_neighbors_list = self.num_neighbors_list).to(self.device)
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
                batch_sampling_result = multihop_sampling(batch_src_index, self.num_neighbors_list, self.adjacency_dict)
                batch_sampling_x = [torch.from_numpy(self.x[idx]).float().to(self.device) for idx in batch_sampling_result]
                batch_train_logits = self.model(batch_sampling_x)
                loss = self.criterion(batch_train_logits, batch_src_label)
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()  
                print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
            self.test()
        print('training complete!')
        
    def test(self):
        test_index = np.where(self.test_mask)[0]
        self.model.eval()
        with torch.no_grad():
            test_sampling_result = multihop_sampling(test_index, self.num_neighbors_list, self.adjacency_dict)
            test_x = [torch.from_numpy(self.x[idx]).float().to(self.device) for idx in test_sampling_result]
            test_logits = self.model(test_x)
            test_label = torch.from_numpy(self.y[test_index]).long().to(self.device)
            predict_y = test_logits.max(1)[1]
            accuarcy = torch.eq(predict_y, test_label).float().mean().item()
            print('-----------------------*-----------------------')
            print("test accuracy: ", accuarcy)
            print('-----------------------*-----------------------')