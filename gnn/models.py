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
from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl._ffi.base import DGLError
from dgl import function as fn
import dgl.nn.pytorch as dglnn
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.nn as nn


# class MLP(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  hidden_dim,
#                  embed_dim,
#                  num_layers: int,
#                  act: str = 'ReLU',
#                  bn: bool = False,
#                  end_up_with_fc=False,
#                  bias=True):
#         super(MLP, self).__init__()
#         self.module_list = []
#         for i in range(num_layers):
#             d_in = input_dim if i == 0 else hidden_dim
#             d_out = embed_dim if i == num_layers - 1 else hidden_dim
#             self.module_list.append(nn.Linear(d_in, d_out, bias=bias))
#             if end_up_with_fc:
#                 continue
#             if bn:
#                 self.module_list.append(nn.BatchNorm1d(d_out))
#             self.module_list.append(getattr(nn, act)(True))
#         self.module_list = nn.Sequential(*self.module_list)

#     def forward(self, x):
#         return self.module_list(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for i in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lins.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.relu(self.lins[i](x))
            x = self.bns[i](x)  # batch norm
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x


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

        # self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
        #                end_up_with_fc=True, act='LeakyReLU')
        self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=3)

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

        # self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
        #                end_up_with_fc=True, act='LeakyReLU')
        self.mlp = MLP(in_feats + hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=3)

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
        # self.mlp = MLP(in_feats + self.heads[-1] * self.hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
        #                end_up_with_fc=True, act='LeakyReLU')
        self.mlp = MLP(in_feats + self.heads[-1] * hidden_dim * n_layers, 2 * n_classes, n_classes, num_layers=3)

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


# ---------------


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads=1,
            feat_drop=0.0,
            attn_drop=0.0,
            edge_drop=0.0,
            negative_slope=0.2,
            use_attn_dst=True,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
            use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GAT(nn.Module):
    def __init__(
            self,
            in_feats,
            n_classes,
            n_hidden,
            n_layers,
            n_heads,
            activation,
            dropout=0.0,
            input_drop=0.0,
            attn_drop=0.0,
            edge_drop=0.0,
            use_attn_dst=True,
            use_symmetric_norm=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # self.mlp = MLP(in_feats, 512, 128, num_layers=3)

        # self.node_encoder = nn.Linear(in_feats, n_hidden)

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats  # modify
            out_hidden = n_hidden  # if i < n_layers - 1 else n_classes
            num_heads = n_heads  # if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        self.pred_linear = nn.Linear(self.n_hidden * self.num_heads, self.n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        # h = self.node_encoder(h)
        # h = F.relu(h, inplace=True)
        h = self.input_drop(h)
        # h = self.mlp(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph[i], h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        # h = h.mean(1)
        # h = self.bias_last(h)
        h = h.flatten(1)
        h = self.pred_linear(h)
        # h = self.mlp(h)

        return F.log_softmax(h, dim=1)
