import gc

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import pickle
import random
import math
import random

probability = 0.5


def read_graphfile(datadir, dataname, max_nodes=None):
    '''
    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset 
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            # zyx here should change to our edge relationship
            if random.random() > 0.5:
                adj_list[graph_indic[e0]].append((e0, e1, 1))
            else:
                adj_list[graph_indic[e0]].append((e0, e1, 2))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        # G=nx.from_edgelist(adj_list[i])
        G = nx.DiGraph()
        for item in adj_list[i]:
            G.add_edge(item[0], item[1], relation=item[2])

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        # print("graph label:")
        # print(graph_labels[i-1])
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.nodes[u]['label'] = node_label_one_hot
                G.nodes[u]['Label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.nodes[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        chance = random.random()

        graphs.append(nx.relabel_nodes(G, mapping))
    # size = math.floor(len(graphs) / 9)
    return graphs


def read_graphfileNew(args, sourceDir, nodeFile, categories, max_nodes=None, hops=None):
    '''
    Returns:
        List of networkx objects with graph and node labels
    '''
    allNodeLabels, allNodeAttributes, allGraphLabels, allEdgeRelations, allNodeToGraph = filesLoader(args, sourceDir,
                                                                                                     nodeFile, hops)

    adj_list = {i: [] for i in range(1, len(allGraphLabels) + 1)}

    for key in allEdgeRelations:
        nodes = key.split('\t')
        a = allNodeToGraph[nodes[0]].split(',')
        b = allNodeToGraph[nodes[1]].split(',')
        for item in [i for i in a if i in b]:
            if args.ablation == '_dis' or args.ablation == '_gcn':
                if allEdgeRelations[key] != 0:
                    adj_list[int(item)].append((nodes[0], nodes[1], 1))
                else:
                    adj_list[int(item)].append((nodes[0], nodes[1], allEdgeRelations[key]))
            else:
                adj_list[int(item)].append((nodes[0], nodes[1], allEdgeRelations[key]))

    uniqueLabels = []
    for label in allGraphLabels.values():
        label = label.split('_')[0]
        if label not in uniqueLabels:
            uniqueLabels.append(label)

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.DiGraph()
        for item in adj_list[i]:
            G.add_edge(item[0], item[1], relation=item[2])
            G.add_edge(item[1], item[0], relation=item[2])

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        G.graph['label'] = allGraphLabels[str(i)]
        for u in G.nodes():
            node_label_one_hot = [0] * categories
            node_label = allNodeLabels[u]
            node_label_one_hot[node_label] = 1
            G.nodes[u]['label'] = node_label_one_hot
            G.nodes[u]['Label'] = node_label_one_hot
            G.nodes[u]['id'] = u
            b = allNodeAttributes[u].copy()
            for i in node_label_one_hot:
                b.append(str(i))
            if args.ablation == '_all':
                b[3] = 0.0
                b[2] = 0.0
                b[4] = 0.0
                b[5] = 0.0
                b[6] = 0.0
            if args.ablation == '_hei':
                b[3] = 0.0
            elif args.ablation == '_area':
                b[2] = 0.0
            elif args.ablation == '_hex':
                b[4] = 0.0
                b[5] = 0.0
                b[6] = 0.0
            elif args.ablation == '_inner':
                b[7] = 0.0
                b[8] = 0.0
                b[9] = 0.0
                b[10] = 0.0
                b[11] = 0.0
                b[12] = 0.0

            G.nodes[u]['feat'] = np.array(b)
        G.graph['feat_dim'] = len(allNodeAttributes[u])

        # relabeling
        mapping = {}
        it = 0

        if float(nx.__version__[0:3]) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1
        graphs.append(nx.relabel_nodes(G, mapping))
    del allNodeLabels, allNodeAttributes, allGraphLabels, allEdgeRelations, allNodeToGraph
    gc.collect()
    return graphs, uniqueLabels


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def filesLoaderWOHex(sourceDir, nodeFile, hops, start, end):
    pois = {}
    with open(nodeFile, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            id = line[0]
            id = id.replace('AOR', '')
            lat = float(line[1])
            lon = float(line[2])
            features = line[3]
            area = line[4]
            height = line[5]
            pois[id] = [lat, lon, features, area, height]

    allNodeLabels = {}
    allNodeAttributes = {}
    allGraphLabels = {}
    allEdgeRelations = {}
    allNodeToGraph = {}

    minLat = 100
    maxLat = -100
    minLon = 100
    maxLon = -100
    minArea = 10000
    maxArea = -10000
    minHei = 1000
    maxHei = -1000
    files = os.listdir(sourceDir)
    files.sort(key=alphanum_key)
    finish = end
    if len(files) - start <= end - start:
        finish = 0
    maxNum = 0
    for idx in range(start, min(end, len(files))):
        # for file in os.listdir(sourceDir):
        file = files[idx]
        fileLabel = file.replace('.txt', '')
        graphLabel = str(len(allGraphLabels) + 1)
        allGraphLabels[str(len(allGraphLabels) + 1)] = fileLabel
        number = 0
        with open(sourceDir + '/' + file, 'r') as f:
            for line in f.readlines():
                line = line.replace('AOR', '')
                line = line.replace('\n', '').split('\t')
                if len(line) > 3:
                    number = number + 1
                    features = pois[line[0]].copy()
                    features[0] = float(line[2])
                    features[1] = float(line[3])
                    category = features[2]
                    category = category.replace('[', '').replace(']', '').split(', ')
                    category = category.index('1')
                    allNodeLabels[line[0]] = category
                    features.append(line[4])
                    features.append(line[5])
                    features.append(line[6])
                    features.append(line[7])
                    features.pop(2)
                    if features[0] < minLat: minLat = features[0]
                    if features[0] > maxLat: maxLat = features[0]
                    if features[1] < minLon: minLon = features[1]
                    if features[1] > maxLon: maxLon = features[1]
                    if float(features[2]) < minArea: minArea = float(features[2])
                    if float(features[2]) > maxArea: maxArea = float(features[2])
                    if float(features[3]) < minHei: minHei = float(features[3])
                    if float(features[3]) > maxHei: maxHei = float(features[3])
                    allNodeAttributes[line[0]] = features
                    if line[0] in allNodeToGraph:
                        allNodeToGraph[line[0]] = allNodeToGraph[line[0]] + ',' + graphLabel
                    else:
                        allNodeToGraph[line[0]] = graphLabel
                else:
                    if float(line[2]) < hops:
                        allEdgeRelations[line[1] + '\t' + line[0]] = int(float(line[2])) + 1
        if number > maxNum:
            maxNum = number
    print('max number is {}'.format(maxNum))
    for key in allNodeAttributes:
        allNodeAttributes[key][0] = (allNodeAttributes[key][0] - minLat) / (maxLat - minLat)
        allNodeAttributes[key][1] = (allNodeAttributes[key][1] - minLon) / (maxLon - minLon)
        allNodeAttributes[key][2] = (float(allNodeAttributes[key][2]) + 1 - minArea) / ((maxArea - minArea) + 0.001)
        allNodeAttributes[key][3] = (float(allNodeAttributes[key][3]) + 1 - minHei) / ((maxHei - minHei) + 0.001)
    return allNodeLabels, allNodeAttributes, allGraphLabels, allEdgeRelations, allNodeToGraph, finish


def filesLoader(args, sourceDir, nodeFile, hops):
    pois = {}
    with open(nodeFile, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            id = line[0]
            id = id.replace('AOR', '')
            lat = float(line[1])
            lon = float(line[2])
            features = line[3]
            area = line[4]
            height = line[5]
            pois[id] = [lat, lon, features, area, height]

    allNodeLabels = {}
    allNodeAttributes = {}
    allGraphLabels = {}
    allEdgeRelations = {}
    allNodeToGraph = {}

    minLat = 100
    maxLat = -100
    minLon = 100
    maxLon = -100
    minArea = 10000
    maxArea = -10000
    minHei = 1000
    maxHei = -1000
    minHx = 1000
    maxHx = -1000
    minHy = 1000
    maxHy = -1000
    minHz = 1000
    maxHz = -1000
    minAge = 1000
    maxAge = -1000
    minInc = 1000
    maxInc = -1000
    files = os.listdir(sourceDir)
    random.shuffle(files)
    maxNum = 0
    for idx in range(int(len(files))):
        # for file in os.listdir(sourceDir):
        file = files[idx]
        fileLabel = file.replace('.txt', '')
        graphLabel = str(len(allGraphLabels) + 1)
        allGraphLabels[str(len(allGraphLabels) + 1)] = fileLabel
        number = 0
        with open(sourceDir + '/' + file, 'r', encoding='unicode_escape') as f:
            for line in f.readlines():
                line = line.replace('AOR', '')
                line = line.replace('\n', '').split('\t')
                if len(line) > 3:
                    number = number + 1
                    features = pois[line[0]].copy()
                    features[0] = float(line[2])
                    features[1] = float(line[3])
                    category = features[2]
                    category = category.replace('[', '').replace(']', '').split(', ')
                    category = category.index('1')
                    allNodeLabels[line[0]] = category
                    hexIdx = line[1].replace('[', '').replace(']', '').split(',')
                    if len(hexIdx) < 3:
                        print(fileLabel)
                        hexIdx = [0, 0, 0]
                    features.append(hexIdx[0])
                    features.append(hexIdx[1])
                    features.append(hexIdx[2])
                    features.append(line[4])
                    features.append(line[5])
                    features.append(line[6])
                    features.append(line[7])
                    features.append(line[8])
                    features.append(line[9])
                    features.pop(2)
                    if features[0] < minLat: minLat = features[0]
                    if features[0] > maxLat: maxLat = features[0]
                    if features[1] < minLon: minLon = features[1]
                    if features[1] > maxLon: maxLon = features[1]
                    if float(features[2]) < minArea: minArea = float(features[2])
                    if float(features[2]) > maxArea: maxArea = float(features[2])
                    if float(features[3]) < minHei: minHei = float(features[3])
                    if float(features[3]) > maxHei: maxHei = float(features[3])
                    if float(features[4]) < minHx: minHx = float(features[4])
                    if float(features[4]) > maxHx: maxHx = float(features[4])
                    if float(features[5]) < minHy: minHy = float(features[5])
                    if float(features[5]) > maxHy: maxHy = float(features[5])
                    if float(features[6]) < minHz: minHz = float(features[6])
                    if float(features[6]) > maxHz: maxHz = float(features[6])
                    if float(features[7]) < minAge: minAge = float(features[7])
                    if float(features[7]) > maxAge: maxAge = float(features[7])
                    if float(features[8]) < minInc: minInc = float(features[8])
                    if float(features[8]) > maxInc: maxInc = float(features[8])
                    allNodeAttributes[line[0]] = features
                    if line[0] in allNodeToGraph:
                        allNodeToGraph[line[0]] = allNodeToGraph[line[0]] + ',' + graphLabel
                    else:
                        allNodeToGraph[line[0]] = graphLabel
                else:
                    if float(line[2]) < hops:
                        allEdgeRelations[line[1] + '\t' + line[0]] = int(float(line[2])) + 1
        if number > maxNum:
            maxNum = number
    print('max number is {}'.format(maxNum))
    for key in allNodeAttributes:
        allNodeAttributes[key][0] = args.weightStart + (allNodeAttributes[key][0] - minLat) / (maxLat - minLat)
        allNodeAttributes[key][1] = args.weightStart + (allNodeAttributes[key][1] - minLon) / (maxLon - minLon)
        allNodeAttributes[key][2] = args.weightStart + (float(allNodeAttributes[key][2]) + 1 - minArea) / (
                (maxArea - minArea) + 0.001)
        allNodeAttributes[key][3] = args.weightStart + (float(allNodeAttributes[key][3]) + 1 - minHei) / (
                (maxHei - minHei) + 0.001)
        allNodeAttributes[key][4] = args.weightStart + (float(allNodeAttributes[key][4]) - minHx) / (
                (maxHx - minHx) + 0.001)
        allNodeAttributes[key][5] = args.weightStart + (float(allNodeAttributes[key][5]) - minHy) / (
                (maxHy - minHy) + 0.001)
        allNodeAttributes[key][6] = args.weightStart + (float(allNodeAttributes[key][6]) - minHz) / (
                (maxHz - minHz) + 0.001)
        allNodeAttributes[key][7] = args.weightStart + (float(allNodeAttributes[key][7]) - minAge) / (
                (maxAge - minAge) + 0.001)
        allNodeAttributes[key][8] = args.weightStart + (float(allNodeAttributes[key][8]) - minInc) / (
                (maxInc - minInc) + 0.001)
    return allNodeLabels, allNodeAttributes, allGraphLabels, allEdgeRelations, allNodeToGraph


def read_supplementarygraph(datadir, task, max_nodes):
    with open(datadir + '/' + task + '.pkl', 'rb') as g:
        graphs = pickle.load(g)

    return graphs
