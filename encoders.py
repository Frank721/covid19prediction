import torch
import torch.nn as nn
# import torch_sparse
from tensorflow import SparseTensor
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def masked_edge_index(edge_index, edge_mask):
    one = torch.ones_like(edge_index)
    zero = torch.zeros_like(edge_index)
    newAdj = torch.where(edge_index == edge_mask, one, zero)
    return newAdj


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, number_relations, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_relations = number_relations
        self.weight = nn.Parameter(torch.FloatTensor(number_relations, input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, relation):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.zeros(x.size(0), x.size(1), self.output_dim, device=x.device)
        for i in range(self.number_relations):
            tmp = masked_edge_index(relation, i + 1)
            h = torch.matmul(tmp, x)
            if self.add_self:
                h += x
            h = torch.matmul(h, self.weight[i])
            y += h
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, number_relations, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None, final_dim='output_dim'):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        # specify whether the last vector is either number of classes or embedding_dim
        self.final_dim = final_dim

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, number_relations, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim
        self.number_relations = number_relations

        # prediction model, output dimension is number of classes
        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        # 2 prediction component
        # first component: map to output_dim
        self.pre_pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, embedding_dim,
                                                     num_aggs=self.num_aggs)
        # second component: map from ouput_dim to label_dim
        self.pred_model = self.build_pred_layers(embedding_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        # mapping model, output dimension is embedding_dim
        self.map_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                embedding_dim, num_aggs=self.num_aggs)

        # a classifier, to be fine-tuned in 2stg+ setting, plugged on top of the final output
        self.map2_model = self.build_pred_layers(pred_input_dim=embedding_dim, pred_hidden_dims=[], label_dim=label_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, number_relations, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, number_relations=number_relations,
                               add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, number_relations=number_relations,
                       add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, number_relations=number_relations,
                              add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                # pred_layers.append(nn.Dropout(0.5))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, relation, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj, relation)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj, relation)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj, relation)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, relation, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj, relation)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        # ==========================
        # if self.embedding_mask is not None:
        #    print(x)
        #    x = x * self.embedding_mask
        #    print(x)`
        # ==========================
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        if self.final_dim == 'pretrain':  # for 2stg+
            out = self.map_model(output)  # will be applied triplet loss
            ypred = self.map2_model(out)
            return ypred, out
        elif self.final_dim != 'output_dim':
            output_vector = self.pre_pred_model(output)
            ypred = self.pred_model(output_vector)
            return output_vector, ypred
        else:  # for 2stg
            ypred = self.map_model(output)
            return output, ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    # when this model is instantiated, embedding_dim = args.output_dim
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, number_relations, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None, final_dim='output_dim'):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, number_relations, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # specify whether the last vector is either number of classes or embedding_dim
        self.final_dim = final_dim
        self.number_relations = number_relations

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, number_relations, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = min(args.max_pooling_num,
                         int(max_num_nodes * assign_ratio))  # zyx the assign dim here is the dimension after pooling.
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, number_relations, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        # 2 prediction component
        # first component: map to output_dim
        self.pre_pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                     embedding_dim, num_aggs=self.num_aggs)
        # second component: map from ouput_dim to label_dim
        self.pred_model = self.build_pred_layers(embedding_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        # final mapping layer
        self.map_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                args.categories, num_aggs=self.num_aggs)

        # a classifier, to be fine-tuned in 2stg+ setting, plugged on top of the final output
        self.map2_model = self.build_pred_layers(pred_input_dim=embedding_dim, pred_hidden_dims=[], label_dim=label_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, relation, batch_num_nodes, **kwargs):
        # if 'assign_x' in kwargs:
        #    x_a = kwargs['assign_x']
        # else:
        x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj, relation,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, relation,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        if self.final_dim == 'pretrain':  # for 2stg+
            out = self.map_model(output)  # will be applied triplet loss
            ypred = self.map2_model(out)
            return ypred, out
        elif self.final_dim != 'output_dim':  # for original
            output_vector = self.pre_pred_model(output)
            ypred = self.pred_model(output_vector)
            return output_vector, ypred
        else:  # for 2stg
            ypred = self.map_model(output)
            return output, ypred

    # this loss is only for original diff-pool setting (final vector = predicted class probabilities)
    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        loss = 0.0
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            # pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1 - adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            return loss + self.link_loss
        return loss


class FinalFC(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], embedding_dim=2):
        super(FinalFC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.final_dim = embedding_dim
        self.bias = True
        self.act = nn.LeakyReLU()

        self.pred_model = self.build_pred_layers(self.input_dim, self.hidden_dims, self.final_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_pred_layers(self, input_dim, pre_hidden_dims, final_dim, num_aggs=1):
        pred_input_dim = input_dim * num_aggs
        if len(pre_hidden_dims) == 0:
            pred_model = nn.Linear(input_dim, final_dim).cuda()
        else:
            pred_layers = []
            pred_layers.append(nn.Linear(input_dim, pre_hidden_dims[0]).cuda())
            # pred_layers.append(nn.Dropout(0.4))
            pred_layers.append(self.act)
            for i in range(len(pre_hidden_dims) - 1):
                pred_layers.append(nn.Linear(pre_hidden_dims[i], pre_hidden_dims[i + 1]).cuda())
                # pred_layers.append(nn.Dropout(0.4))
                pred_layers.append(self.act)

            pred_layers.append(nn.Linear(pre_hidden_dims[-1], final_dim).cuda())
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self, x):
        out = self.pred_model(x)
        return out


class Ensemble(nn.Module):
    def __init__(self, model1, model2):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, adj=None, batch_num_nodes=None, assign_x=None):
        if adj != None:
            a_h0 = x
            a_adj = adj
        else:
            a_adj = Variable(torch.Tensor([x.graph['adj']]), requires_grad=False).cuda()
            a_h0 = Variable(torch.Tensor([x.graph['feats']]), requires_grad=False).cuda()
        if batch_num_nodes == None:
            a_batch_num_nodes = np.array([x.graph['num_nodes']])
        if assign_x == None:
            assign_x = Variable(torch.Tensor(x.graph['assign_feats']), requires_grad=False).cuda()

        x1 = self.model1(a_h0, a_adj, batch_num_nodes, assign_x=assign_x)

        x2 = self.model2(x1.cuda())

        return x2
