import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import random
import networkx as nx
import torch.nn as nn
from sklearn.metrics import f1_score
import scipy.sparse as sp
import itertools

'''
    Implementing ClusterGraph Algorithm using backend PyTorch
    @Author: Shuaiheng Xiao 

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
   
   num_clusters -> int
               number of subgraph 
   
   cluster_method -> string
                {'random', 'metis'}, if 'random', randomly cut graph into n clusters
                                    if 'metis', import Pymetis do the clustering part.
                                    default: 'metis'
   
   epochs -> int
               default: 20
               
   lr -> float
               learning rate default: 0.01
         
   residual_block -> boolean
               model including residual block or not, default: True
               
   save_path -> string
               saving path, default: None
               
   v1:
   1. updata metis clustering method
   2. adding adjacency normalizaion L=D^-0.5 * (A+I) * D^-0.5
   
'''
class preprocessing():
    
    def __init__(self,
                 node_df,
                 edge_df,
                 num_clusters,
                 cluster_method):
        
        
        assert all(col in node_df.columns for col in ['cust_id','is_driver','is_reported'])
        assert all(col in edge_df.columns for col in ['cust_id','opp_id'])
        assert type(num_clusters) == int
        
        self.node_df = node_df
        self.edge_df = edge_df
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.data_dict = {}

    def run(self):
        
        self.delete_nodes()
        graph = self.build_graph(self.edge_df)
        if self.cluster_method == 'metis':
            clusters, cluster_membership = self.metis_clustering()
        elif self.cluster_method == 'random':
            clusters, cluster_membership = self.random_clustering(graph)
        
        self.adjacency = self.build_adjacency()
        self.adjacency = self.normalization()
        self.adjacency = self.adjacency2tensor()
        self.feats_name = list(set(self.node_df.columns) - set(['cust_id','is_driver','is_reported']))
        features = self.node_df[self.feats_name]
        features = torch.from_numpy(np.array(features)).float()
        features = torch.sparse.mm(self.adjacency,features)
        self.node_df = pd.concat([self.node_df[['cust_id','is_driver','is_reported']],pd.DataFrame(data = np.array(features),columns = self.feats_name)], axis = 1)
        
        self.data_dict['sg_nodes'], self.data_dict['sg_edges'], self.data_dict['sg_train_nodes'], self.data_dict['sg_test_nodes'], self.data_dict['sg_train_features'], self.data_dict['sg_test_features'], self.data_dict['sg_train_targets'], self.data_dict['sg_test_targets'] = self.build_membership_dict(graph,clusters, cluster_membership)
        
        
        return graph, self.data_dict
    
    
    def delete_nodes(self):
        
        # node_lookup: store node index
        node_lookup = pd.DataFrame({'node': self.node_df.index}, index=self.node_df.cust_id)

        # delete no-edge-node 
        diff_node = list(set(self.node_df['cust_id'])-(set(self.node_df['cust_id']) - set(self.edge_df['cust_id']) - set(self.edge_df['opp_id'])))

        self.node_df = self.node_df.iloc[node_lookup.iloc[diff_node]['node']].reset_index(drop=True)

        
    def build_graph(self, edge_df):
        # build up graph using networkx

        graph = nx.from_edgelist([(cust, opp) for cust, opp in zip(edge_df['cust_id'], edge_df['opp_id'])])
        
        return graph

    def metis_clustering(self):

        import pymetis
        clusters = [cluster for cluster in range(self.num_clusters)]
        node_lookup = pd.DataFrame({'node': self.node_df.index, }, index = self.node_df.cust_id)
        self.adjacency_dict = defaultdict(list)
        for cust, opp in zip(self.edge_df['cust_id'], self.edge_df['opp_id']):
            self.adjacency_dict[node_lookup.loc[cust]['node']].append(node_lookup.loc[opp]['node'])
        adjacency_list = []
        for node in list(self.node_df['cust_id']):
            adjacency_list.append(self.adjacency_dict[node])
        _, membership= pymetis.part_graph(self.num_clusters, adjacency_list)
        cluster_membership = {}
        for node, member in zip(list(self.node_df['cust_id']), membership):
            cluster_membership[node] = member
        return clusters, cluster_membership


    def random_clustering(self, graph):
        # random_clustering
        clusters = [cluster for cluster in range(self.num_clusters)]
        cluster_membership = {node: random.choice(clusters) for node in graph.nodes()}
        
        node_lookup = pd.DataFrame({'node': self.node_df.index, }, index = self.node_df.cust_id)
        self.adjacency_dict = defaultdict(list)
        for cust, opp in zip(self.edge_df['cust_id'], self.edge_df['opp_id']):
            self.adjacency_dict[node_lookup.loc[cust]['node']].append(node_lookup.loc[opp]['node'])
        
        return clusters, cluster_membership
    
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
        num_nodes, input_dim = self.node_df.shape
        input_dim -= 3
        indices = torch.from_numpy(np.asarray([self.adjacency.row, 
                                               self.adjacency.col]).astype('int64')).long()
        values = torch.from_numpy(self.adjacency.data.astype(np.float32))
        tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                                    (num_nodes, num_nodes))
        return tensor_adjacency
    
    
    
    def build_membership_dict(self, graph, clusters, cluster_membership):
        
        # build-up membership dict
        sg_nodes = {}
        sg_edges = {}
        sg_train_nodes = {}
        sg_test_nodes = {}
        sg_train_features = {}
        sg_test_features = {}
        sg_train_targets = {}
        sg_test_targets = {}

        for cluster in clusters:

            #print(cluster)
            subgraph = graph.subgraph([node for node in sorted(graph.nodes()) if cluster_membership[node] == cluster])
            sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]

            mapper = {node: i for i, node in enumerate(sorted(sg_nodes[cluster]))}
            sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]

            sg_train_nodes[cluster] = [node for node in self.node_df[self.node_df['is_driver'] == True]['cust_id'] if node in sg_nodes[cluster]]
            sg_test_nodes[cluster] = [node for node in self.node_df[self.node_df['is_driver'] == False]['cust_id'] if node in sg_nodes[cluster]]

            sg_test_nodes[cluster] = sorted(sg_test_nodes[cluster])
            sg_train_nodes[cluster] = sorted(sg_train_nodes[cluster])
            
            
            sg_train_features[cluster] = pd.concat([self.node_df[(self.node_df['cust_id'] == cust)&(self.node_df['is_driver'] == True)][self.feats_name] for cust in sg_nodes[cluster]],axis = 0)
            sg_test_features[cluster] = pd.concat([self.node_df[(self.node_df['cust_id'] == cust)&(self.node_df['is_driver'] == False)][self.feats_name] for cust in sg_nodes[cluster]],axis = 0)
            
            sg_train_targets[cluster] = pd.concat([self.node_df[(self.node_df['cust_id'] == cust)&(self.node_df['is_driver'] == True)][['is_reported']] * 1 for cust in sg_nodes[cluster]],axis = 0)
            sg_test_targets[cluster] = pd.concat([self.node_df[(self.node_df['cust_id'] == cust)&(self.node_df['is_driver'] == False)][['is_reported']] * 1 for cust in sg_nodes[cluster]],axis = 0)
          
        return sg_nodes, sg_edges, sg_train_nodes, sg_test_nodes, sg_train_features, sg_test_features, sg_train_targets, sg_test_targets
    
    

class GCN(torch.nn.Module):
    '''
    basic graph convolutional layer
    '''
    def __init__(self, input_dim, output_dim, activation = torch.nn.functional.relu):
        
        super(GCN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        self.weight1 = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        #self.weight2 = nn.Parameter(torch.Tensor(self.input_dim * 2, self.output_dim))
        self.bn1 = nn.BatchNorm1d(self.output_dim)
        #self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        
        torch.nn.init.kaiming_uniform_(self.weight1)
        #torch.nn.init.kaiming_uniform_(self.weight2.weight)
        
    def forward(self,features):
        
        output = self.bn1(self.activation(torch.matmul(features,self.weight1)))
        #print(output.shape)
        
        return output

class residual_block(torch.nn.Module):
    '''
    basic residual block with convolutional layer
    
    '''
    
    def __init__(self,input_dim, output_dim,):
        
        super(residual_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gcn = GCN(self.input_dim, self.output_dim)
        self.linear_1 = nn.Linear(output_dim,output_dim)
        self.bn1 = nn.BatchNorm1d(self.output_dim)
        self.droput = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(output_dim,output_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
     
        torch.nn.init.xavier_uniform_(self.linear_1.weight,
                                          gain = torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.linear_2.weight,
                                          gain = torch.nn.init.calculate_gain('relu'))
        
    def forward(self, features):
        output = self.gcn(features)
        dummy_1 = output
        output = self.linear_1(output)
        output = torch.nn.functional.relu(output)
        output = self.bn1(output)
        output = self.droput(output)
        output = torch.add(output,dummy_1)
        dummy_2 = output
        output = self.linear_2(output)
        output = self.bn2(output)
        output = torch.nn.functional.relu(output)
        output = torch.add(output,dummy_2)
        
        return output
    
class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
    
class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, hidden_dims, input_channels, output_channels,residual_block):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features. 
        """
        super(StackedGCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_block = residual_block
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        self.all_dims = [self.input_channels] + self.hidden_dims + [self.output_channels]
        for i, _ in enumerate(self.all_dims[:-1]):
            if residual_block:
                self.layers.append(residual_block(self.all_dims[i],self.all_dims[i+1]))
            else:
                self.layers.append(GCN(self.all_dims[i],self.all_dims[i+1]))
        self.layers = ListModule(*self.layers)

    def forward(self, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        #print(self.layers)
        for i, _ in enumerate(self.all_dims[:-2]):
            features = torch.nn.functional.relu(self.layers[i](features))
            if i>1:
                features = torch.nn.functional.dropout(features,0.3)
        features = self.layers[i+1](features)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions

    
class run_model():
    
    def __init__(self,
                 node_df,
                 edge_df,
                 num_clusters,
                 hidden_dims,
                 cluster_method = 'metis',
                 epochs = 20,
                 lr = 0.01,
                 residual_block = True,
                 device = 'cpu',
                 save_path = None):
        
        self.node_df = node_df
        self.edge_df = edge_df
        self.num_clusters = num_clusters
        self.clusters = [cluster for cluster in range(self.num_clusters)]
        self.hidden_dims = hidden_dims
        self.cluster_method = cluster_method
        self.epochs = epochs
        self.lr = lr
        self.residual_block = residual_block
        self.device = device
        self.save_path = save_path
        
        print('-----------------------*-----------------------')
        print('data preprocessing ..\n')
        pre_model = preprocessing(self.node_df, self.edge_df, self.num_clusters, self.cluster_method)
        self.graph, self.data_dict = pre_model.run()
        print('preprocessing completed!')
        print('-----------------------*-----------------------')
        
        self.feature_count = int(self.data_dict['sg_train_features'][0].shape[1])  #input dim
        self.class_count = int(np.max(self.data_dict['sg_train_targets'][0]) + 1)  # output dim
        print('-----------------------*-----------------------')
        print('model construction\n')
        self.creat_model()
        self.ToTensor()
        print(self.model)
        print('-----------------------*-----------------------')
        
        
    def creat_model(self):
        
        self.model = StackedGCN(self.hidden_dims,self.feature_count,self.class_count,self.residual_block)
        self.model = self.model.to(self.device)
    
    def ToTensor(self):
        
        for cluster in self.clusters:
            self.data_dict['sg_nodes'][cluster] = torch.LongTensor(self.data_dict['sg_nodes'][cluster])
            self.data_dict['sg_edges'][cluster] = torch.LongTensor(self.data_dict['sg_edges'][cluster]).t()
            self.data_dict['sg_train_nodes'][cluster] = torch.LongTensor(self.data_dict['sg_train_nodes'][cluster])
            self.data_dict['sg_test_nodes'][cluster] = torch.LongTensor(self.data_dict['sg_test_nodes'][cluster])
            
            self.data_dict['sg_train_features'][cluster] = torch.FloatTensor(np.array(self.data_dict['sg_train_features'][cluster]))
            self.data_dict['sg_train_targets'][cluster] = torch.LongTensor(np.array(self.data_dict['sg_train_targets'][cluster]))
            self.data_dict['sg_test_features'][cluster] = torch.FloatTensor(np.array(self.data_dict['sg_test_features'][cluster]))
            self.data_dict['sg_test_targets'][cluster] = torch.LongTensor(np.array(self.data_dict['sg_test_targets'][cluster]))
        
    
    def do_forward_pass(self, cluster):
        
        edges = self.data_dict['sg_edges'][cluster].to(self.device)
        macro_nodes = self.data_dict['sg_nodes'][cluster].to(self.device)
        train_nodes = self.data_dict['sg_train_nodes'][cluster].to(self.device)
        train_features = self.data_dict['sg_train_features'][cluster].to(self.device)
        train_target = self.data_dict['sg_train_targets'][cluster].to(self.device).squeeze()
        predictions = self.model(train_features)
#         print('predictions ',predictions.shape)
#         print('train_target ', train_target.shape)
#         print('train_nodes', train_nodes.shape)
        #average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], train_target[train_nodes])
        average_loss = torch.nn.functional.nll_loss(predictions, train_target)
        node_count = train_nodes.shape[0]

        return average_loss, node_count
    
    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.data_dict['sg_edges'][cluster].to(self.device)
        macro_nodes = self.data_dict['sg_nodes'][cluster].to(self.device)
        test_nodes = self.data_dict['sg_test_nodes'][cluster].to(self.device)
        test_features = self.data_dict['sg_test_features'][cluster].to(self.device)
        test_target = self.data_dict['sg_test_targets'][cluster].to(self.device).squeeze()

        prediction = self.model(test_features)
        
        return prediction, test_target
    
    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        
        return average_loss
    
    def train(self):
        """
        Training a model.
        """
        print("Training started.\n")
        print('training through {}\n'.format(self.device))
        #epochs = trange(self.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            random.shuffle(self.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
                print("Epoch {:03d} Cluster {:03d} Loss: {:.4f}".format(epoch, cluster, average_loss))
            #epochs.set_description("Train Loss: %g" % round(average_loss,4))
            self.test()
        print('training complete!')
        if self.save_path:
            torch.save(self.model, self.save_path)
    
    
    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="micro")
        print('-----------------------*-----------------------')
        print("test accuracy: ", sum(self.predictions == self.targets)/len(self.targets))
        print("\nF-1 score: {:.4f}".format(score))
        print('-----------------------*-----------------------')