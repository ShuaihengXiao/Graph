from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import scipy.sparse as sp
import itertools

from tensorboardX import SummaryWriter
writer = SummaryWriter('/Users/shuaihengxiao/Desktop/graphSAGE_v0/runs/exp1')



"""
   Applying GraphSAGE Using Backend PyTorch
   @author: Shuaiheng Xiao
   
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
               
      
   
  
   v2 update:
   1. fix dimension dismatch when aggr_hidden_method = concat 
   2. fix when aggr_neighbor_method = max cause wrong return type problem
   
   
   v1 update:
   1. include residual block 
   2. apply batch normalization 
"""

def data_format_process(node_df,edge_df):
    """
    output
    x: node features (array: float)
    y: label         (array: float)
    adjacency_dict: a dict store neighbor node's info  {node_index:[neighbor1,neighbor2..]}
    train_mask: mask for providing training dataset (list: bool)
    test_mask: mask for providing testing dataset (list: bool)
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
    
    # convert to Array
    x = node_df[set(node_df) - {'cust_id', 'is_driver', 'is_reported'}].to_numpy()
    y = node_df.is_reported.to_numpy() * 1
    
    # mask conf
    train_mask = node_df.is_driver.to_numpy()
    test_mask = ~train_mask
    
    return x, y, adjacency_dict, train_mask, test_mask


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
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
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
        init.kaiming_uniform_(self.weight)
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
            
        #print('input_dim shape',self.input_dim)
        #print('output_dim shape',self.output_dim)
        #print('aggr_neighbor shape',aggr_neighbor.shape)
        #print('self.weight shape',self.weight.shape)
        
        
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
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
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
        self.dropout1 = nn.Dropout(0.4)
        #self.dropout2 = nn.Dropout(0.2)
        
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
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
#         print('self.input_dim shape',self.input_dim)
#         print('self.hidden_dim shape', self.hidden_dim)
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        #print('neighbor_node_features shape',neighbor_node_features.shape) #torch.Size([16, 10, 120])
        #print('self.wight shape', self.weight.shape)                       #torch.Size([120, 128])
        #print('neighbor_hidden shape', neighbor_hidden.shape)              #torch.Size([16, 128])
        #print('aggr_neighbor shape', aggr_neighbor.shape)                  #torch.Size([16, 120])
        
        #print('src_node_features', src_node_features.shape)                #torch.Size([16, 120])
        #print('self_hidden', self_hidden.shape)                            #torch.Size([16, 128])
        
 
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
            hidden = self.dropout1(hidden)
            hidden = F.relu(self.linear_2(hidden))
            hidden = self.bn_2(hidden)
            #hidden = self.dropout2(hidden)
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
                 data,
                 hidden_dim = [128, 7],
                 num_neighbors_list = [10, 10],
                 aggr_neighbor_method = 'mean',
                 aggr_hidden_method = "sum",
                 batch_size = 16,
                 epochs = 20,
                 num_batch_per_epoch = 20,
                 lr = 0.01,
                 residual_block = True,
                 save_path = None):
        
#         assert isinstance(node_df, pd.DataFrame)
#         assert isinstance(edge_df, pd.DataFrame)
        assert isinstance(hidden_dim, list)
        assert isinstance(num_neighbors_list, list)
        assert len(hidden_dim) == len(num_neighbors_list)
        
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
        
        print('data preprocessing..')
        self.x,self.y,self.adjacency_dict,self.train_mask,self.test_mask = data.x,data.y,data.adjacency_dict,data.test_mask,data.train_mask
        
        self.x = self.x / self.x.sum(1, keepdims=True)
        self.adjacency = self.build_adjacency()
        self.adjacency = self.normalization()
        self.adjacency = self.adjacency2tensor()
        self.x = torch.from_numpy(self.x).float().to(self.device)
        self.x = torch.sparse.mm(self.adjacency,self.x)
        print('data preprocessing complete!')
        print('-----------------------*-----------------------')
        
        print('after filtering single nodes')
        print('num of train instances: {}'.format(sum(self.train_mask)))
        print('num of test instances: {}'.format(sum(self.test_mask)))
        print('-----------------------*-----------------------')
        
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
        
    def build_adjacency(self):
        """
           build adjacency matrix according to adjacency dictionary
        """
        edge_index = []
        num_nodes = len(self.adjacency_dict)
        for src, dst in self.adjacency_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), 
                                   (edge_index[:, 0], edge_index[:, 1])),
                    shape=(num_nodes, num_nodes), dtype="float32")
        
        return adjacency  
    
    def normalization(self):
        """
            calculate L=D^-0.5 * (A+I) * D^-0.5
        """
        self.adjacency += sp.eye(self.adjacency.shape[0])    
        degree = np.array(self.adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        
        return d_hat.dot(self.adjacency).dot(d_hat).tocoo()
    
    def adjacency2tensor(self):
        """
            convert numpy.array adjacency matrix to torch.tensor
        """
        num_nodes, input_dim = self.x.shape
        indices = torch.from_numpy(np.asarray([self.adjacency.row, 
                                               self.adjacency.col]).astype('int64')).long()
        values = torch.from_numpy(self.adjacency.data.astype(np.float32))
        tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                                    (num_nodes, num_nodes))
        return tensor_adjacency

    def train(self):
        
        
        train_index = np.where(self.train_mask)[0]
        train_label = self.y
        maximum_accu = 0
        print('model training..')
        print('training through {}'.format(self.device))
        self.model.train()
        for e in range(self.epochs):
            for batch in range(self.num_batch_per_epoch):
                batch_src_index = np.random.choice(train_index, size=(self.batch_size,))
                batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(self.device)
                batch_sampling_result = multihop_sampling(batch_src_index, self.num_neighbors_list, self.adjacency_dict)
                batch_sampling_x = [self.x[idx] for idx in batch_sampling_result]
                batch_train_logits = self.model(batch_sampling_x).squeeze(1)
                loss = self.criterion(batch_train_logits, batch_src_label)
                self.optimizer.zero_grad()
                loss.backward()  
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 5.0)
                self.optimizer.step()  
                print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
            accuracy = self.test()
            writer.add_scalar('test_accu', accuracy, e) 
            maximum_accu = max(maximum_accu,accuracy)
            
            writer.add_scalar('batch_loss', loss.item(), e) 
            for name, param in self.model.named_parameters():
                
                writer.add_histogram(
                    name, param.clone().data.numpy(), e)


            
        print('training complete!')
        print('best accuracy: ',maximum_accu)
        writer.close()
        if self.save_path:
            torch.save(model,path)
            print('model saved!')
        
    def test(self):
        test_index = np.where(self.test_mask)[0]
        self.model.eval()

        with torch.no_grad():
            test_sampling_result = multihop_sampling(test_index, self.num_neighbors_list, self.adjacency_dict)
            test_x = [self.x[idx] for idx in test_sampling_result]
            test_logits = self.model(test_x).squeeze(1)
            test_label = torch.from_numpy(self.y[test_index]).long().to(self.device)
            predict_y = test_logits.max(1)[1]
            #predict_y = (test_logits > 0.5).float()
            accuarcy = torch.eq(predict_y, test_label).float().mean().item()
            
            print('-----------------------*-----------------------')
            print("test accuracy: ", accuarcy)
            
            print('-----------------------*-----------------------')
        return accuarcy