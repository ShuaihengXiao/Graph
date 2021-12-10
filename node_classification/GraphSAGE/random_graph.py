#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd


def random_graph_lcd(
    node_n,
    edge_n,
    seed_n,
    directed=False,
    random_weight=False,
    random_retain=False,
    random_seed=0
):
    
    """generate node,edge dataframe for lcd test
    node_n: nums of nodes
    edge_n: nums of edges
    seed_n: nums of black seeds"""
    
    np.random.seed(random_seed)

    nodes = np.array(range(node_n))

    full_edges = np.array(range(node_n * node_n))
    edges = np.random.choice(
        full_edges[full_edges // node_n > full_edges % node_n],
        size=edge_n
    )

    seeds = np.random.choice(nodes, size=seed_n)

    if random_weight:
        edge_weights = np.random.random(edge_n)
    else:
        edge_weights = np.ones(edge_n)

    if random_retain:
        node_retain = np.random.random(node_n) * 0.3
    else:
        node_retain = np.full(node_n, 0.15)

    if not directed:
        edges = np.concatenate((
            edges,
            edges // node_n + edges % node_n * node_n,
        ))

        edge_weights = np.concatenate((
            edge_weights,
            edge_weights,
        ))

    node_dataframe = pd.DataFrame({
        'cust_id': nodes,
        'is_reported': np.in1d(nodes, seeds),
        'weight': np.array([
            edge_weights[edges % node_n == i].sum()
            for i in nodes
        ]),
        'retain': node_retain,
    })

    edge_dataframe = pd.DataFrame({
        'cust_id': edges % node_n,
        'opp_id': edges // node_n,
        'weight': edge_weights,
    }).sort_values('cust_id')

    return node_dataframe, edge_dataframe






def random_graph_gcn(node_n,
                     edge_n,
                     report_rate = 0.5,
                     driver_rate = 0.8,
                     nums_features = 10,
                     random_weight = True,
                     random_seed = 0):
    """
    generate node,edge dataframe for gcn test
    report_rate: rate of neg samples
    nums_features: nums feats wanna be created
    """
    np.random.seed(random_seed)
    nodes = np.array(range(node_n))

    full_edges = np.array(range(node_n * node_n))
    edges = np.random.choice(
        full_edges[full_edges // node_n > full_edges % node_n],
        size=edge_n)
    seeds = np.random.choice(nodes, size = int(node_n*report_rate))
    seeds_driver = np.random.choice(nodes, size = int(node_n*driver_rate))
    edges = np.concatenate((
            edges,
            edges // node_n + edges % node_n * node_n, ))
    
    
    node_dataframe = pd.DataFrame({
        'cust_id': nodes,
        'is_driver': np.in1d(nodes, seeds_driver),
        'is_reported': np.in1d(nodes, seeds),
    })
    
    if nums_features:
        features_name_lst = ['feat_' + str(i) for i in range(nums_features)]
        for name in features_name_lst:
            node_dataframe[name] = np.random.randn(node_n)
    if random_weight:
        edge_weights = np.random.random(edge_n)
    else:
        edge_weights = np.ones(edge_n)
    edge_weights = np.concatenate((
            edge_weights,
            edge_weights,
        ))
    edge_dataframe = pd.DataFrame({
        'cust_id': edges % node_n,
        'opp_id': edges // node_n,
        'weight': edge_weights,
    }).sort_values('cust_id')

    return node_dataframe, edge_dataframe


# In[ ]:




