import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler


def prepare_val_data(graphs, args, val_idx, max_nodes=0):
    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx + 1) * val_size:]
    val_graphs = graphs[val_idx * val_size: (val_idx + 1) * val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch

    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    print("feat dim")
    print(dataset_sampler.feat_dim)
    return train_dataset_loader, val_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


# split train, val, test sets: for original differential pooling setting (each train, val, test is a data loader)
def split_train_val_normal(graphs, args, val_test_idx, max_nodes, feat):
    # split train, val, test

    ## if there is a validation set: 80% train, 10% val, 10% test

    if args.val == True:
        val_test_size = len(graphs) // 5
        train_graphs = graphs[:val_test_idx * val_test_size]
        if val_test_idx < 4:
            train_graphs = train_graphs + graphs[(val_test_idx + 1) * val_test_size:]
        val_test_graphs = graphs[val_test_idx * val_test_size: (val_test_idx + 1) * val_test_size]
        val_size = len(val_test_graphs) // 2
        val_graphs = val_test_graphs[:val_size]
        test_graphs = val_test_graphs[val_size:]

    ## if there is no validation set: 90% train, 10% test
    else:
        test_idx = val_test_idx
        test_size = len(graphs) // 10
        train_graphs = graphs[:test_idx * test_size]
        if test_idx < 9:
            train_graphs = train_graphs + graphs[(test_idx + 1) * test_size:]
        test_graphs = graphs[test_idx * test_size: (test_idx + 1) * test_size]

    # train set loader
    print(len(train_graphs))
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes, features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    # test set loader
    testset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes, features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        testset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    if args.val:
        valset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes, features=args.feature_type)
        val_dataset_loader = torch.utils.data.DataLoader(
            valset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    else:
        val_dataset_loader = test_dataset_loader

    # print("feat dim")
    # print(dataset_sampler.feat_dim)
    return train_dataset_loader, test_dataset_loader, val_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


# split train, val, test sets: for triplet train setting (each train, val, test is a dictionary, keys are the classes, values are arrays of graphs)
def train_only(train_graphs, args, max_nodes, feat, samplesEach):
    num_classes = args.num_classes

    # shuffle the dataset
    random.shuffle(train_graphs)

    train_graphs_dict = dict()

    node_list = list(train_graphs[0].nodes)
    representative_node = node_list[0]

    feat_dim = train_graphs[0].nodes[representative_node]['feat'].shape[0]
    assign_feat_dim = feat_dim

    for train_graph in train_graphs:
        num_nodes = train_graph.number_of_nodes()
        # label
        label = train_graph.graph['label']

        # adj
        adj = np.array(nx.to_numpy_matrix(train_graph))
        adj_padded = np.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        train_graph.graph['adj'] = adj_padded

        # relations
        adj = nx.to_dict_of_dicts(train_graph)
        adj_padded = np.zeros((max_nodes, max_nodes))
        for key1, values in adj.items():
            for key2, value in values.items():
                adj_padded[key1, key2] = value['relation']
        train_graph.graph['relation'] = adj_padded

        # feats
        f = np.zeros((max_nodes, feat_dim), dtype=float)
        for i, u in enumerate(train_graph.nodes()):
            if args.feature_type == 'node-label':
                f[i, :] = train_graph.nodes[u]['feat']
            else:
                # f[i,:] = (train_graph.nodes[u]['feat'].data).cpu().numpy()
                f[i, :] = train_graph.nodes[u]['feat'].data
        train_graph.graph['feats'] = f

        # num_nodes
        train_graph.graph['num_nodes'] = num_nodes
        # assign feats
        train_graph.graph['assign_feats'] = f
        train_graphs_dict[label] = train_graph

    return train_graphs_dict, max_nodes, feat_dim, assign_feat_dim


# split train, val, test sets: for triplet train setting (each train, val, test is a dictionary, keys are the classes, values are arrays of graphs)
def split_train_val(train_graphs, test_graphs, args, val_test_idx, max_nodes, feat, samplesEach):
    num_classes = args.num_classes
    # shuffle the dataset
    random.shuffle(train_graphs)

    train_graphs_dict = dict()
    test_graphs_dict = dict()
    val_graphs_dict = dict()

    node_list = list(train_graphs[0].nodes)
    representative_node = node_list[0]

    feat_dim = train_graphs[0].nodes[representative_node]['feat'].shape[0]
    assign_feat_dim = feat_dim

    for train_graph in train_graphs:
        num_nodes = train_graph.number_of_nodes()
        # label
        label = train_graph.graph['label']

        # adj
        adj = np.array(nx.to_numpy_matrix(train_graph))
        adj_padded = np.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        train_graph.graph['adj'] = adj_padded

        # relations
        adj = nx.to_dict_of_dicts(train_graph)
        adj_padded = np.zeros((max_nodes, max_nodes))
        for key1, values in adj.items():
            for key2, value in values.items():
                adj_padded[key1, key2] = value['relation']
        train_graph.graph['relation'] = adj_padded

        # feats
        f = np.zeros((max_nodes, feat_dim), dtype=float)
        for i, u in enumerate(train_graph.nodes()):
            if args.feature_type == 'node-label':
                f[i, :] = train_graph.nodes[u]['feat']
            else:
                # f[i,:] = (train_graph.nodes[u]['feat'].data).cpu().numpy()
                f[i, :] = train_graph.nodes[u]['feat'].data
        train_graph.graph['feats'] = f

        # num_nodes
        train_graph.graph['num_nodes'] = num_nodes
        # assign feats
        train_graph.graph['assign_feats'] = f
        train_graphs_dict[label] = train_graph

    for test_graph in test_graphs:

        num_nodes = test_graph.number_of_nodes()
        # label
        label = (test_graph.graph['label'])

        # adj
        adj = np.array(nx.to_numpy_matrix(test_graph))
        adj_padded = np.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        test_graph.graph['adj'] = adj_padded

        # relations
        adj = nx.to_dict_of_dicts(train_graph)
        adj_padded = np.zeros((max_nodes, max_nodes))
        for key1, values in adj.items():
            for key2, value in values.items():
                adj_padded[key1, key2] = value['relation']
        test_graph.graph['relation'] = adj_padded

        # feats
        f = np.zeros((max_nodes, feat_dim), dtype=float)
        for i, u in enumerate(test_graph.nodes()):
            if args.feature_type == 'node-label':
                f[i, :] = test_graph.nodes[u]['feat']
            else:
                # f[i,:] = (test_graph.nodes[u]['feat'].data).cpu().numpy()
                f[i, :] = test_graph.nodes[u]['feat'].data

        test_graph.graph['feats'] = f

        # num_nodes
        test_graph.graph['num_nodes'] = num_nodes

        # assign feats
        test_graph.graph['assign_feats'] = f

        test_graphs_dict[label] = (test_graph)

    if len(test_graphs_dict) < 1:
        test_graphs_dict = None

    return train_graphs_dict, test_graphs_dict, val_graphs_dict, \
           max_nodes, feat_dim, assign_feat_dim


def obtain_search(search_graphs, args, max_nodes):
    num_classes = args.num_classes

    # shuffle the dataset
    # random.shuffle(search_graphs)

    search_graphs_dict = dict()
    search_graphs_positive_dict = dict()

    node_list = list(search_graphs[0].nodes)
    representative_node = node_list[0]

    feat_dim = search_graphs[0].nodes[representative_node]['feat'].shape[0]

    for search_graph in search_graphs:
        num_nodes = search_graph.number_of_nodes()
        # label
        label = search_graph.graph['label']

        # adj
        adj = np.array(nx.to_numpy_matrix(search_graph))
        adj_padded = np.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        search_graph.graph['adj'] = adj_padded

        # relations
        adj = nx.to_dict_of_dicts(search_graph)
        adj_padded = np.zeros((max_nodes, max_nodes))
        for key1, values in adj.items():
            for key2, value in values.items():
                adj_padded[key1, key2] = value['relation']
        search_graph.graph['relation'] = adj_padded

        # feats
        f = np.zeros((max_nodes, feat_dim), dtype=float)
        for i, u in enumerate(search_graph.nodes()):
            if args.feature_type == 'node-label':
                f[i, :] = search_graph.nodes[u]['feat']
            else:
                # f[i,:] = (train_graph.nodes[u]['feat'].data).cpu().numpy()
                f[i, :] = search_graph.nodes[u]['feat'].data
        search_graph.graph['feats'] = f

        # num_nodes
        search_graph.graph['num_nodes'] = num_nodes
        # assign feats
        search_graph.graph['assign_feats'] = f
        if 'p' in label:
            search_graphs_positive_dict[label] = search_graph
        else:
            search_graphs_dict[label] = search_graph

    return search_graphs_dict, search_graphs_positive_dict
