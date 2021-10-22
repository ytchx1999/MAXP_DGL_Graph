import argparse
from dgl.data import CitationGraphDataset

import os
import dgl
import pickle
import numpy as np
import torch as th


def load_dgl_graph(base_path):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############', flush=True)
    print(graph, flush=True)

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


def load_graph(name):
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph('../dataset')
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    nodes = graph.nodes()
    nodes_train, y_train = nodes[train_nid], labels[train_nid]
    nodes_val, y_val = nodes[val_nid], labels[val_nid]
    nodes_train = th.cat([nodes_train, nodes_val], dim=0)
    y_train = th.cat([y_train, y_val], dim=0)
    eval_set = [(nodes_train, y_train), (nodes_val, y_val)]

    return graph, eval_set


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='Node2vec')
    parser.add_argument('--dataset', type=str, default='cora')
    # 'train' for training node2vec model, 'time' for testing speed of random walk
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=50)
    parser.add_argument('--p', type=float, default=0.25)  # 0.25
    parser.add_argument('--q', type=float, default=4)  # 4
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    graph, eval_set = load_graph(args.dataset)
    print('done!', flush=True)
