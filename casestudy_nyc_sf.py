import torch
import gc
from torch.autograd import Variable
import argparse
import os
import shutil
import numpy as np
import cross_val
import encoders
import load_data
import warnings

warnings.filterwarnings('ignore')


# evaluate based on MLP
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio * 100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def embed(anchor, model):
    a_adj = Variable(torch.Tensor([anchor.graph['adj']]), requires_grad=False).cuda()
    a_relation = Variable(torch.Tensor([anchor.graph['relation']]), requires_grad=False).cuda()
    a_h0 = Variable(torch.Tensor([anchor.graph['feats']]), requires_grad=False).cuda()
    a_batch_num_nodes = np.array([anchor.graph['num_nodes']])
    a_assign_input = Variable(torch.Tensor(anchor.graph['assign_feats']), requires_grad=False).cuda()
    _, embed_a = model(a_h0, a_adj, a_relation, a_batch_num_nodes, assign_x=a_assign_input)
    return embed_a.cpu().detach().numpy()


def compute_similarity(emb1, emb2):
    minDis = 10000000000
    for i in range(len(emb1)):
        for j in range(len(emb2)):
            dis = 0
            for k in range(len(emb1[i])):
                dis += np.sum(np.abs(emb1[i][k] - emb2[j][k]))
            minDis = min(minDis, dis / 1000)
    return minDis


def search(dataset, searchTarget_dataset, model, modelDir):
    resMatrix = np.zeros((len(dataset), len(searchTarget_dataset)))
    keys = list(dataset.keys())
    total = list(searchTarget_dataset.keys())
    embedList = []
    embedTotalList = []
    for i in range(len(keys)):
        idx = keys.index(str(i))
        anchor = dataset[keys[idx]]
        embed_a = embed(anchor, model)
        embedList.append(embed_a)
    for i in range(len(total)):
        idx = total.index(str(i))
        anchor = searchTarget_dataset[total[idx]]
        embed_t = embed(anchor, model)
        embedTotalList.append(embed_t)
    for i in range(len(keys)):
        for j in range(len(total)):
            dist_p = compute_similarity(embedList[i], embedTotalList[j])
            resMatrix[i][j] = dist_p

    res = np.zeros((len(dataset), len(searchTarget_dataset)))
    for i in range(len(resMatrix)):
        res[i] = np.argsort(resMatrix[i], axis=-1, kind='quicksort', order=None)

    modzcats = []
    with open('SF/names.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            modzcats.append(line.replace('\n', ''))
    with open('NYC/casestudy_nycsf{}.txt'.format(modelDir), 'w') as file_proceed:
        for i in range(len(res)):
            for j in range(len(res[i])):
                idx = res[i][j]
                file_proceed.write(str(modzcats[int(idx)]) + '\t')
            file_proceed.write('\n')
    return


def mutual_node(source, end):
    mutual = 0
    for outter in source:
        for inner in end:
            outterId = outter['id']
            innerId = inner['id']
            if outterId == innerId:
                mutual = 1
                break
        if mutual == 1:
            break
    return mutual


def benchmark_task_val(args, ratio, ablation, modelDir):
    # do the search Evaluate
    assign_input_dim = args.input_dim
    model = encoders.SoftPoolingGcnEncoder(
        args.max_nodes,
        args.input_dim, args.hidden_dim, args.output_dim, args.number_relations, args.num_classes, args.num_gc_layers,
        args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        assign_input_dim=assign_input_dim, final_dim='output_dim').cuda()
    model.load_state_dict(torch.load(
        './SF/model{}/net_params{}_{}_{}.pth'.format(modelDir, ablation, args.max_pooling_num, args.number_relations)))
    [searchGraphs, _] = load_data.read_graphfileNew(args, './NYC/samples/search',
                                                    './NYC/NYC_w.txt',
                                                    args.categories, max_nodes=args.max_nodes,
                                                    hops=args.number_relations)
    [searchTargetGraphs, _] = load_data.read_graphfileNew(args, './SF/samples/search',
                                                          './SF/SF_w.txt',
                                                          args.categories, max_nodes=args.max_nodes,
                                                          hops=args.number_relations)
    print('search graphs loaded')
    [search_dataset, _] = cross_val.obtain_search(searchGraphs, args, max_nodes=args.max_nodes)
    [searchTarget_dataset, _] = cross_val.obtain_search(searchTargetGraphs, args, max_nodes=args.max_nodes)
    del searchGraphs
    gc.collect()
    search(search_dataset, searchTarget_dataset, model, modelDir)
    del search_dataset, searchTarget_dataset
    gc.collect()


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset', default='DD')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')
    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign_ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--task', dest='task',
                        help='Whether attack task or ppi task')
    parser.add_argument('--top', dest='top', type=int,
                        help='Top 2 or top 3 for attack data')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=True,
                        help='Whether link prediction side objective is used')
    # whether to have a validation sets
    parser.add_argument('--val', dest='val', type=bool,
                        help='Whether to have a validation set')
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.', default=20)
    parser.add_argument('--iterations', dest='num_iterations', type=int,
                        help='Number of iterations in benchmark_test_val')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature_type', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension', default=32)
    parser.add_argument('--samplesEach', dest='samplesEach', type=int,
                        help='samples for each item', default=5)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension', default=32)
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension', default=32)
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes', default=2)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--buckSize', dest='buckSize', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--buck', dest='buck', type=int,
                        help='Maximum bucks, should be multiple of 5.')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign', default='base')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')
    parser.add_argument('--alpha', dest='alpha', type=float,
                        help='margin alpha in the triplet loss', default=1.5)
    parser.add_argument('--triplet', dest='triplet',
                        help='how triplet is sampled')
    parser.add_argument('--weightStart', dest='weightStart', type=float,
                        help='weightStart.')
    parser.add_argument('--l2_regularize', dest='l2_regularize', type=float,
                        help='coefficient of L2 regularization, to control parameters')
    parser.add_argument('--l1_regularize', dest='l1_regularize', type=float,
                        help='coefficient of L1 reguarization, to enforce sparsity')
    parser.add_argument('--w', dest='w', type=float,
                        help='weight of covariance matrix regularization')
    parser.add_argument('--gamma', dest='gamma', type=float,
                        help='decay rate of variance in the covariance matrix (refer to draft)')
    parser.add_argument('--categories', dest='categories', type=int,
                        help='number of categories')
    parser.add_argument('--max_pooling_num', dest='max_pooling_num', type=int,
                        help='number of max_pooling_num')
    parser.add_argument('--noise', dest='noise', type=float,
                        help='noise')
    parser.add_argument('--ablation', dest='ablation',
                        help='ablation', default='')
    parser.add_argument('--number_relations', dest='number_relations', type=int,
                        help='neighbor ranges [<200,200-400,400-800,800-1600,1600-] to 0-5.', default=3)
    parser.set_defaults(task='ppi',
                        datadir='SF',
                        logdir='log',
                        val=True,
                        dataset='syn1v2',
                        cuda='0',
                        feature_type='node-feat',
                        categories=30,
                        lr=0.05,
                        clip=2.0,
                        batch_size=1,
                        buck=500,
                        weightStart=0.2,
                        buckSize=300,
                        num_epochs=5,
                        num_iterations=1,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=43,
                        hidden_dim=20,
                        output_dim=10,
                        num_classes=10,
                        number_relations=3,
                        num_gc_layers=2,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=1,
                        num_pool=1,
                        samplesEach=5,
                        alpha=1.5,
                        max_pooling_num=30,
                        max_nodes=500,
                        noise=0.1,
                        ablation='',
                        triplet='random',
                        l1_regularize=0,
                        l2_regularize=0,
                        w=0,
                        gamma=2
                        )
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    modelDir = '_inner'
    prog_args.ablation = '_inner'
    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
    if prog_args.task == 'ppi':
        print('ablation study: {}  assign ratio: {}   noise: {}'.format(prog_args.ablation, prog_args.assign_ratio,
                                                                        prog_args.noise))
        benchmark_task_val(prog_args, prog_args.noise, prog_args.ablation, modelDir)


if __name__ == "__main__":
    main()
