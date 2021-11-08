import argparse
import copy
import os
from dgl import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model import MLP, MLPLinear, CorrectAndSmooth
from load_dataset import load_dgl_graph, time_diff
from load_dataset import prepare_data
import pandas as pd
import pickle
from tqdm import tqdm
import time


import argparse
import glob
import os

import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

from load_dataset import load_dgl_graph
from model import CorrectAndSmooth
import gc
import pickle
import pandas as pd
import time
import dgl

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_output_files(output_path):
    outputs = glob.glob(output_path)
    print(f"Detect {len(outputs)} model output files")
    assert len(outputs) > 0
    probs_list = []
    for out in outputs:
        # probs = torch.zeros(size=(n_nodes, n_classes), device="cpu")
        # probs[tr_va_te_nid] = torch.load(out, map_location="cpu")
        probs = torch.load(out, map_location="cpu")
        probs_list.append(probs)
        # mx_diff = (out_probs[-1].sum(dim=-1) - 1).abs().max()
        # if mx_diff > 1e-1:
        #     print(f'Max difference: {mx_diff}')
        #     print("model output doesn't seem to sum to 1. Did you remember to exp() if your model outputs log_softmax()?")
        #     raise Exception
    return probs_list


def generate_preds_path(args):
    path = os.path.join('../dataset',
                        f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}" +
                        f"_K_{args.K}_label_K_{args.label_K}_probs_seed_*_stage_*.pt")  # {args.stage}
    return path


def main():
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph('../dataset')
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    graph = graph.to(device)
    labels = labels.to(device)
    train_nid = torch.from_numpy(train_nid).to(device)
    val_nid = torch.from_numpy(val_nid).to(device)
    test_nid = torch.from_numpy(test_nid).to(device)
    node_feat = node_feat.to(device)

    n_features = node_feat.size()[-1]
    n_classes = 23

    # if args.pretrain:
    print('---------- Before ----------', flush=True)
    # model.load_state_dict(torch.load(f'base/{args.dataset}-{args.model}.pt'))
    # model.eval()

    # y_soft = torch.rand(labels.shape[0], 23)
    y_soft_gat = torch.load('../dataset/y_soft.pt', map_location='cpu')

    # use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)
    # preds_path = generate_preds_path(args)
    # print(preds_path)
    # probs_list = load_output_files(preds_path)
    # probs0 = probs_list[0]
    # probs1 = probs_list[1]
    # y_soft_sagn0 = torch.zeros((labels.shape[0], probs0.shape[1]))
    # y_soft_sagn0[tr_va_te_nid] = probs0
    # y_soft_sagn1 = torch.zeros((labels.shape[0], probs1.shape[1]))
    # y_soft_sagn1[tr_va_te_nid] = probs1

    y_soft_gamlp = []

    for i in range(args.num_ensemble):
        if os.path.exists(f'../dataset/gamlp_{i}.pt'):
            y_soft_gamlp.append(torch.zeros((labels.shape[0], y_soft_gat.shape[1])))
            y_soft_gamlp[i][tr_va_te_nid] = torch.load(f'../dataset/gamlp_{i}.pt', map_location='cpu')
        # if os.path.exists(f'../dataset/tmp-gamlp/gamlp_{i}.pt'):
        #     y_soft_gamlp.append(torch.zeros((labels.shape[0], y_soft_gat.shape[1])))
        #     y_soft_gamlp[i][tr_va_te_nid] = torch.load(f'../dataset/gamlp_{i}.pt', map_location='cpu')
        else:
            break

    # y_soft_sage = None
    # y_soft_conv = None
    # if os.path.exists('../dataset/y_soft_sage.pt'):
    #     y_soft_sage = torch.load('../dataset/y_soft_sage.pt', map_location='cpu')
    # if os.path.exists('../dataset/y_soft_sage.pt'):
    #     y_soft_conv = torch.load('../dataset/y_soft_conv.pt', map_location='cpu')
    # if y_soft_sage != None and y_soft_conv != None:
    #     y_soft = 0.6 * y_soft + 0.2 * y_soft_sage + 0.2 * y_soft_conv

    y_pred_gamlp = []
    val_acc_gamlp = []

    for i in range(len(y_soft_gamlp)):
        y_pred_gamlp.append(None)
        val_acc_gamlp.append(None)

    for i in range(len(y_soft_gamlp)):
        y_soft_gamlp[i] = y_soft_gamlp[i].softmax(dim=-1).to(device)
        y_pred_gamlp[i] = y_soft_gamlp[i].argmax(dim=-1)
        val_acc_gamlp[i] = torch.sum(y_pred_gamlp[i][val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        print(f'Pre valid acc: {val_acc_gamlp[i]:.4f}', flush=True)

    print('---------- Correct & Smoothing ----------', flush=True)
    cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                          correction_alpha=args.correction_alpha,
                          correction_adj=args.correction_adj,
                          num_smoothing_layers=args.num_smoothing_layers,
                          smoothing_alpha=args.smoothing_alpha,
                          smoothing_adj=args.smoothing_adj,
                          autoscale=args.autoscale,
                          scale=args.scale)

    if args.all_train:
        mask_idx = torch.cat([train_nid, val_nid])
    else:
        mask_idx = train_nid

    for i in range(len(y_soft_gamlp)):
        y_soft_gamlp[i] = cs.correct(graph, y_soft_gamlp[i], labels[mask_idx], mask_idx)
        y_soft_gamlp[i] = cs.smooth(graph, y_soft_gamlp[i], labels[mask_idx], mask_idx)
        y_pred_gamlp[i] = y_soft_gamlp[i].argmax(dim=-1)
        val_acc_gamlp[i] = torch.sum(y_pred_gamlp[i][val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        print(f'Valid acc: {val_acc_gamlp[i]:.4f}', flush=True)

    # y_soft = cs.correct(graph, y_soft, labels[mask_idx], mask_idx)
    # y_soft = cs.smooth(graph, y_soft, labels[mask_idx], mask_idx)
    # y_pred = y_soft.argmax(dim=-1)
    # val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    # print(f'Final valid acc: {val_acc:.4f}', flush=True)

    y_soft = 0
    w = [0.2] * len(y_soft_gamlp)
    # w = [
    #     0.1, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.5, 0.5, 0.1,
    #     0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1
    # ]
    for i in range(len(y_soft_gamlp)):
        y_soft += (w[i] * y_soft_gamlp[i])

    # y_soft = 0.2 * y_soft_gamlp + 0.2 * y_soft_gamlp1 + 0.2 * y_soft_gamlp2 + 0.2 * y_soft_gamlp3 + 0.2 * y_soft_gamlp4 \
    #     + 0.2 * y_soft_gamlp5 + 0.2 * y_soft_gamlp6 + 0.2 * y_soft_gamlp7 + 0.2 * y_soft_gamlp8 + 0.2 * y_soft_gamlp9
    # + 0.2 * y_soft_sagn0 + 0.2 * y_soft_sagn1 + \
    # 0.2 * y_soft_sage + 0.1 * y_soft_gat
    # y_soft = y_soft_gamlp

    y_soft = y_soft.softmax(dim=-1).to(device)
    y_pred = y_soft.argmax(dim=-1)
    val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    print(f'Pre valid acc: {val_acc:.4f}', flush=True)

    # test
    with open(os.path.join('../dataset/test_id_dict.pkl'), 'rb') as f:
        test_id_dict = pickle.load(f)
    submit = pd.read_csv('../dataset/sample_submission_for_validation.csv')
    with open(os.path.join('../dataset/csv_idx_map.pkl'), 'rb') as f:
        idx_map = pickle.load(f)
    # save results
    test_pred_list = y_pred[test_nid]
    for i, id in tqdm(enumerate(test_nid)):
        paper_id = test_id_dict[id.item()]
        label = chr(int(test_pred_list[i].item() + 65))

        # csv_index = submit[submit['id'] == paper_id].index.tolist()[0]
        if paper_id in idx_map:
            csv_index = idx_map[paper_id]
            submit['label'][csv_index] = label

    if not os.path.exists('../outputs'):
        os.makedirs('../outputs', exist_ok=True)
    submit.to_csv(os.path.join('../outputs/', f'submit_gamlp_cs_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

    print("Done!", flush=True)


if __name__ == '__main__':
    """
    Correct & Smoothing Hyperparameters
    """
    parser = argparse.ArgumentParser(description='Base predictor(C&S)')

    # Dataset
    parser.add_argument('--gpu', type=int, default=1, help='1 for cpu')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products'])
    # Base predictor
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'linear'])
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hid-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    # extra options for gat
    parser.add_argument('--n-heads', type=int, default=3)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    # C & S
    parser.add_argument('--pretrain', action='store_true', help='Whether to perform C & S')
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0.979)  # 0.979
    parser.add_argument('--correction-adj', type=str, default='DAD')
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0.756)  # 0.756
    parser.add_argument('--smoothing-adj', type=str, default='DAD')
    parser.add_argument('--autoscale', action='store_true')
    parser.add_argument('--scale', type=float, default=1.5)
    parser.add_argument('--all_train', action='store_true')
    parser.add_argument('--num-ensemble', type=int, default=10)
    parser.add_argument("--K", type=int, default=5,
                        help="Maximum hop for feature propagation")
    parser.add_argument("--label-K", type=int, default=9,
                        help="Maximum hop for label propagation (in SLE)")
    parser.add_argument("--stage", type=int, default=2,
                        help="Which stage in SLE to postprocess")
    parser.add_argument("--use-labels", action="store_true",
                        help="Whether to enhance base model with a label model")
    parser.add_argument("--avoid-features", action="store_true",
                        help="Whether to ignore node features (only useful when using labels)")

    args = parser.parse_args()
    print(args, flush=True)

    main()
