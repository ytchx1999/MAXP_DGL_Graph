# -*- coding:utf-8 -*-

"""
    Utilities to handel graph data
"""

import os
import dgl
import pickle
import numpy as np
import torch as th
# from ogb.nodeproppred import DglNodePropPredDataset


def load_dgl_graph(base_path):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############', flush=True)
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx']
    val_label_idx = label_data['val_label_idx']
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################', flush=True)
    print('Total labels (including not labeled): {}'.format(labels.shape[0]), flush=True)
    print('               Training label number: {}'.format(tr_label_idx.shape[0]), flush=True)
    print('             Validation label number: {}'.format(val_label_idx.shape[0]), flush=True)
    print('                   Test label number: {}'.format(test_label_idx.shape[0]), flush=True)

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############', flush=True)
    print('Node\'s feature shape:{}'.format(node_feat.shape), flush=True)

    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)
