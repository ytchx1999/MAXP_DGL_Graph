import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn
import torch
from layer import *


class R_GAMLP(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, act="relu", pre_process=False, residual=False, pre_dropout=False, bns=False, fine_tune=None):
        super(R_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.residual = residual
        self.pre_dropout = pre_dropout
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

        if fine_tune != None:
            for param in self.prelu.parameters():
                param.requires_grad = False
            for param in self.lr_att.parameters():
                param.requires_grad = False
            for param in self.process.parameters():
                param.requires_grad = False
            for param in self.res_fc.parameters():
                param.requires_grad = False

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        return right_1


class JK_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, act, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(JK_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        self.pre_dropout = pre_dropout
        if pre_process:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*hidden, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*nfeat, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(nfeat + hidden, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        return right_1


class JK_GAMLP_RLU(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(JK_GAMLP_RLU, self).__init__()
        self.num_hops = num_hops
        self.pre_dropout = pre_dropout
        self.prelu = nn.PReLU()
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        if pre_process:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*hidden, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*nfeat, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(nfeat + hidden, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)
        self.pre_process = pre_process
        self.label_fc = FeedForwardNet(
            nclass, hidden, nclass, n_layers_3, dropout)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.residual = residual

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_emb):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        right_1 += self.label_fc(self.label_drop(label_emb))
        return right_1


class R_GAMLP_RLU(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(R_GAMLP_RLU, self).__init__()
        self.num_hops = num_hops
        self.pre_dropout = pre_dropout
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.label_drop = nn.Dropout(label_drop)
        self.residual = residual
        self.label_fc = FeedForwardNet(
            nclass, hidden, nclass, n_layers_3, dropout)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_emb):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        right_1 += self.label_fc(self.label_drop(label_emb))
        return right_1


# adapt from https://github.com/facebookresearch/NARS/blob/main/model.py
class WeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(
                torch.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feat_list):  # feat_list k (N,S,D)
        new_feats = []
        for feats, weight in zip(feat_list, self.agg_feats):
            new_feats.append(
                (feats * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


class NARS_JK_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(NARS_JK_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = JK_GAMLP(nfeat, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                              alpha, n_layers_1, n_layers_2, pre_process, residual, pre_dropout, bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1


class NARS_R_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(NARS_R_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = R_GAMLP(nfeat, hidden, nclass, num_hops, dropout, input_drop,
                             attn_drop, alpha, n_layers_1, n_layers_2, pre_process, residual, pre_dropout, bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1


class NARS_JK_GAMLP_RLU(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(NARS_JK_GAMLP_RLU, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = JK_GAMLP_RLU(nfeat, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                  label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process, residual, pre_dropout, bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1


class NARS_R_GAMLP_RLU(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(NARS_R_GAMLP_RLU, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = R_GAMLP_RLU(nfeat, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                 label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process, residual, pre_dropout, bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

        self.linears.append(nn.Linear(hid_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = linear(x)
            x = F.relu(x, inplace=True)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return F.log_softmax(x, dim=-1)


class LabelPropagation(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """

    def __init__(self, num_layers, alpha, adj='DAD'):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj

    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)

            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.adj == 'DAD' else -1).to(labels.device).unsqueeze(1)

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                if self.adj in ['DAD', 'AD']:
                    y = norm * y

                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')

                if self.adj in ['DAD', 'DA']:
                    y = y * norm

                y = post_step(last + y)

            return y


class CorrectAndSmooth(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_
    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """

    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 correction_adj,
                 num_smoothing_layers,
                 smoothing_alpha,
                 smoothing_adj,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()

        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers,
                                      correction_alpha,
                                      correction_adj)
        self.prop2 = LabelPropagation(num_smoothing_layers,
                                      smoothing_alpha,
                                      smoothing_adj)

    def correct(self, g, y_soft, y_true, mask):
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x

                smoothed_error = self.prop1(g, error, post_step=fix_input)

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, g, y_soft, y_true, mask):
        with g.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            y_soft[mask] = y_true
            return self.prop2(g, y_soft)
