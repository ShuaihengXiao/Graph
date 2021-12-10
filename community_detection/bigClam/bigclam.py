import random
import numpy as np
import networkx as nx

'''
    implemention of BigCLAM by 
    https://github.com/benedekrozemberczki/karateclub/blob/aa8c2e779734b2e8ec47716aa97d810f0a401b8e/karateclub/community_detection/overlapping/bigclam.py
    
    
'''
def ensure_integrity(graph: nx.classes.graph.Graph):
        """Ensure walk traversal conditions."""
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph
    
def check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."
        
def check_graph(graph: nx.classes.graph.Graph):
        """Check the Karate Club assumptions about the graph."""
        check_indexing(graph)
        graph = ensure_integrity(graph)

        return graph
    
    
class BigClam():
    """
    Args:
        dimensions (int): Number of embedding dimensions. Default 8.
        iterations (int): Number of training iterations. Default 50.
        learning_rate (float): Gradient ascent learning rate. Default is 0.005.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 8,
        iterations: int = 50,
        learning_rate: int = 0.005,
        seed: int = 42,
    ):
        self.dimensions = dimensions
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.seed = seed

    def _initialize_features(self, number_of_nodes):
        """
        Creating the community embedding and gradient sum.
        Arg types:
            * **number_of_nodes** *(int)* - The number of nodes in the graph.
        """
        self._embedding = np.random.uniform(0, 1, (number_of_nodes, self.dimensions))
        self._global_features = np.sum(self._embedding, axis=0)

    def _calculate_gradient(self, node_feature, neb_features):
        """
        Calculating the feature gradient.
        Arg types:
            * **node_feature** *(Numpy array)* - The node representation.
            * **neb_features** *(Numpy array)* - The representation of node neighbours.
        """
        raw_scores = node_feature.dot(neb_features.T)
        raw_scores = np.clip(raw_scores, -15, 15)
        scores = np.exp(-raw_scores) / (1 - np.exp(-raw_scores))
        scores = scores.reshape(-1, 1)
        neb_grad = np.sum(scores * neb_features, axis=0)
        without_grad = (
            self._global_features - node_feature - np.sum(neb_features, axis=0)
        )
        grad = neb_grad - without_grad
        return grad

    def _do_updates(self, node, gradient, node_feature):
        """
        Updating the embedding and the feature sum.
        Arg types:
            * **node** *(int)* - The node identifier.
            * **gradient** *(Numpy array)* - The gradient of the node representation.
            * **node_feature** *(Numpy array)* - The node representation.
        """
        self._embedding[node] = self._embedding[node] + self.learning_rate * gradient
        self._embedding[node] = np.clip(self._embedding[node], 0.00001, 10)
        self._global_features = (
            self._global_features - node_feature + self._embedding[node]
        )

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        indices = np.argmax(self._embedding, axis=1)
        memberships = {i: membership for i, membership in enumerate(indices)}
        return memberships

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = self._embedding
        return embedding

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a BigClam clustering model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """

        graph = check_graph(graph)
        number_of_nodes = graph.number_of_nodes()
        self._initialize_features(number_of_nodes)
        nodes = [node for node in graph.nodes()]
        for i in range(self.iterations):
            random.shuffle(nodes)
            for node in nodes:
                nebs = [neb for neb in graph.neighbors(node)]
                neb_features = self._embedding[nebs, :]
                node_feature = self._embedding[node, :]
                gradient = self._calculate_gradient(node_feature, neb_features)
                self._do_updates(node, gradient, node_feature)