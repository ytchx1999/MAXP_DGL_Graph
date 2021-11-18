import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from heteo_data import load_data, read_relation_subsets, gen_rel_subset_feature, preprocess_features
import torch.nn.functional as F
import gc
import os
import pickle


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

    labels = torch.from_numpy(label_data['label'])
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
    node_feat = torch.from_numpy(features).float()
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


def prepare_label_emb(args, g, labels, n_classes, train_idx, valid_idx, test_idx, label_teacher_emb=None):
    if args.dataset == 'ogbn-mag':
        target_type_id = g.get_ntype_id("paper")
        homo_g = dgl.to_homogeneous(g, ndata=["feat"])
        homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
        homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
        feat = g.ndata['feat']['paper']
    print(n_classes, flush=True)
    print(labels.shape[0], flush=True)
    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)
    else:
        print("use teacher label", flush=True)
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[valid_idx] = label_teacher_emb[len(
            train_idx):len(train_idx)+len(valid_idx)]
        y[test_idx] = label_teacher_emb[len(
            train_idx)+len(valid_idx):len(train_idx)+len(valid_idx)+len(test_idx)]
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)

    if args.dataset == 'ogbn-mag':
        target_mask = homo_g.ndata["target_mask"]
        target_ids = homo_g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_label_emb = torch.zeros((len(homo_g.ndata["feat"]),) + y.shape[1:],
                                    dtype=y.dtype, device=y.device)
        new_label_emb[target_mask] = y[target_ids]
        y = new_label_emb
        g = homo_g
    del labels
    gc.collect()
    for hop in range(args.label_num_hops):
        y = neighbor_average_labels(g, y.to(torch.float), args)
        gc.collect()
    if args.dataset == "ogbn-mag":
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = torch.zeros((num_target,) + y.shape[1:],
                              dtype=y.dtype, device=y.device)
        new_res[target_ids] = y[target_mask]
        y = new_res
    res = y
    return torch.cat([res[train_idx], res[valid_idx], res[test_idx]], dim=0)


def neighbor_average_labels(g, feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels", flush=True)
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    '''if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)

        # error
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                               dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res'''

    return feat


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats", flush=True)
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res


def batched_acc(labels, pred):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)


def get_evaluator(dataset):
    dataset = dataset.lower()
    if dataset.startswith("oag"):
        return batched_ndcg_mrr
    else:
        return batched_acc


def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
#    if dataset=='ogbn-mag':
#        return batched_acc
#    else:
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def get_maxp_evaluator():
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
#    if dataset=='ogbn-mag':
#        return batched_acc
#    else:
    # evaluator = Evaluator(name=dataset)
    return lambda preds, labels: ((preds.view(-1, 1).detach().cpu() == labels.view(-1, 1).detach().cpu()).sum().item() / labels.detach().cpu().shape[0])


def load_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    if name not in ["ogbn-products", "ogbn-mag", "ogbn-papers100M", "maxp"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    if args.dataset == 'maxp':  # maxp
        g, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph('../dataset')
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)

        # 419 no feature nodes -- mean aggr of there neighbor
        node_feat_new = node_feat.clone()
        diff_node_feat = torch.zeros((419, node_feat.shape[1]))
        node_feat_new[-419:, :] = diff_node_feat

        with g.local_scope():
            g.ndata['h'] = node_feat_new
            g.update_all(fn.copy_u('h', 'm'),
                         fn.mean('m', 'h'))
            node_feat_new = g.ndata.pop('h')

        node_feat[-419:, :] = node_feat_new[-419:, :].clone()
        print("mean aggr done!", flush=True)

        print("Use node2vec embedding...", flush=True)
        emb = torch.load('../dataset/emb.pt', map_location='cpu')
        emb.requires_grad = False
        node_feat = torch.cat([node_feat, emb], dim=1)

        # y_soft_gat = torch.load('../dataset/y_soft_gat.pt', map_location='cpu')
        # node_feat = torch.cat([node_feat, y_soft_gat], dim=1)

        g.ndata["feat"] = node_feat.float()
        g.ndata["labels"] = labels
        train_nid = torch.from_numpy(train_nid)
        val_nid = torch.from_numpy(val_nid)
        test_nid = torch.from_numpy(test_nid)
        n_classes = 23
        labels = labels.squeeze()
        evaluator = get_maxp_evaluator()
        # evaluator = None
        # g = g.to(device)
    else:
        dataset = DglNodePropPredDataset(name=name, root=args.root)
        splitted_idx = dataset.get_idx_split()

    if name == "ogbn-products":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        g.ndata["labels"] = labels
        g.ndata['feat'] = g.ndata['feat'].float()
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-mag":
        data = load_data(device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid = data
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n", flush=True)

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args, teacher_probs):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    data = load_dataset(args.dataset, device, args)

    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    if args.dataset == 'ogbn-products' or args.dataset == 'maxp':
        feats = neighbor_average_features(g, args)
        in_feats = feats[0].shape[1]
    elif args.dataset == 'ogbn-mag':
        rel_subsets = read_relation_subsets(args.use_relation_subsets)

        with torch.no_grad():
            feats = preprocess_features(g, rel_subsets, args, device)
            print("Done preprocessing", flush=True)
        _, num_feats, in_feats = feats[0].shape
    elif args.dataset == 'ogbn-papers100M':
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        feat = g.ndata.pop('feat')
    gc.collect()
    label_emb = None
    if args.use_rlu:
        label_emb = prepare_label_emb(args, g, labels, n_classes, train_nid, val_nid, test_nid, teacher_probs)
    # move to device
    if args.dataset == 'ogbn-papers100M':

        feats = []
        for i in range(args.num_hops+1):
            feats.append(torch.load(f"/data2/zwt/ogbn_papers100M/feat/papers100m_feat_{i}.pt"))
        in_feats = feats[0].shape[1]
        '''
        g.ndata['feat']=feat
        feats=neighbor_average_features(g,args)
        in_feats=feats[0].shape[1]
        
        for i, x in enumerate(feats):
            feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
        '''
    else:
        for i, x in enumerate(feats):
            feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    labels = labels.to(device).to(torch.long)
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]), in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator, label_emb
