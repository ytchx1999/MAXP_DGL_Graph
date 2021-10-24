# -*- coding:utf-8 -*-

# Author:james Zhang

"""
    Three common GNN models.
"""

import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(nn, act)(True))
        self.module_list = nn.Sequential(*self.module_list)

    def forward(self, x):
        return self.module_list(x)


class GraphSageModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0):
        super(GraphSageModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()
        self.bns = thnn.ModuleList()
        self.res_linears = nn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean'))
        # aggregator_type = 'pool'))
        self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
        self.res_linears.append(torch.nn.Linear(in_feats, hidden_dim))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean'))
            # aggregator_type='pool'))
            self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
            self.res_linears.append(torch.nn.Identity())
        self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean'))
        self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
        self.res_linears.append(torch.nn.Identity())
        # aggregator_type = 'pool'))

        self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')

    # def forward(self, blocks):
    #     collect = []
    #     h = blocks[0].srcdata['feat']
    #     h = self.dropout(h)
    #     num_output_nodes = blocks[-1].num_dst_nodes()
    #     collect.append(h[:num_output_nodes])
    #     for l, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h_res = h[:block.num_dst_nodes()]
    #         h = layer(block, h)
    #         h = self.bns[l](h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #         collect.append(h[:num_output_nodes])
    #         h += self.res_linears[l](h_res)
    #     return self.mlp(torch.cat(collect, -1))

    def forward(self, blocks, features):
        h = features
        h = F.dropout(h, p=0.1, training=self.training)
        collect = []
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            h = self.bns[l](h)
            h = self.activation(h)
            h = self.dropout(h)

            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)

        return self.mlp(torch.cat(collect, -1))


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()
        self.bns = thnn.ModuleList()
        self.res_linears = nn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation, ))
        self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
        self.res_linears.append(torch.nn.Linear(in_feats, hidden_dim))

        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation))
            self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
            self.res_linears.append(torch.nn.Identity())

        self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation))
        self.bns.append(thnn.BatchNorm1d(self.hidden_dim))
        self.res_linears.append(torch.nn.Identity())

        self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')

    def forward(self, blocks, features):
        # h = features
        h = features
        h = F.dropout(h, p=0.1, training=self.training)
        collect = []
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])

        # for l, (layer, block) in enumerate(zip(self.layers, blocks)):
        #     h = layer(block, h)
        #     if l != len(self.layers) - 1:
        #         h = self.dropout(h)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            h = self.bns[l](h)
            h = self.activation(h)
            h = self.dropout(h)

            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)

        return self.mlp(torch.cat(collect, -1))
        # return h


class GraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop
                 ):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation

        self.layers = thnn.ModuleList()
        self.norms = nn.ModuleList()

        # build multiple layers
        # self.layers.append(dglnn.GATConv(in_feats=self.in_feats,
        #                                  out_feats=self.hidden_dim,
        #                                  num_heads=self.heads[0],
        #                                  feat_drop=self.feat_dropout,
        #                                  attn_drop=self.attn_dropout,
        #                                  activation=self.activation))

        self.node_encoder = nn.Linear(in_feats, hidden_dim)

        for l in range(self.n_layers):
            in_hidden = self.heads[l - 1] * self.hidden_dim if l > 0 else self.hidden_dim
            out_hidden = self.hidden_dim
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=in_hidden,
                                             out_feats=out_hidden,
                                             num_heads=self.heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             activation=self.activation))
            self.norms.append(nn.BatchNorm1d(self.heads[l] * out_hidden))

        # self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[-2],
        #                                  out_feats=self.n_classes,
        #                                  num_heads=self.heads[-1],
        #                                  feat_drop=self.feat_dropout,
        #                                  attn_drop=self.attn_dropout,
        #                                  activation=None))

        self.pred_linear = nn.Linear(self.heads[-1] * self.hidden_dim, self.n_classes)
        self.mlp = MLP(in_feats + self.heads[-1] * self.hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')

        self.input_drop = nn.Dropout(p=0.1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, blocks, features):
        h = features
        h = self.input_drop(h)
        h = self.node_encoder(h)
        # collect = []
        # num_output_nodes = blocks[-1].num_dst_nodes()
        # collect.append(h[:num_output_nodes])

        h_last = None

        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[l](h)
            h = self.activation(h)
            h = self.dropout(h)

            # collect.append(h[:num_output_nodes])

        # logits = self.layers[-1](blocks[-1], h).mean(1)
        h = self.pred_linear(h)
        # h = self.mlp(torch.cat(collect, -1))

        return h
