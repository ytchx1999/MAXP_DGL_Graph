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

    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)

    y_soft_kfold = []

    idx = 0
    for seed in range(args.num_seed):
        for k in range(args.kfold):
            if idx >= 20:
                break
            if os.path.exists(f'../dataset/gamlp_{k}fold_seed{seed}.pt'):
                y_kfold = torch.load(f'../dataset/gamlp_{k}fold_seed{seed}.pt', map_location='cpu')
                y_soft_kfold.append(torch.zeros((labels.shape[0], y_kfold.shape[1])))
                y_soft_kfold[idx][tr_va_te_nid] = y_kfold
                idx += 1
            else:
                break

    y_soft_gamlp = []

    for i in range(args.num_ensemble):
        if os.path.exists(f'../dataset/gamlp_{i}.pt'):
            y_gamlp = torch.load(f'../dataset/gamlp_{i}.pt', map_location='cpu')
            y_soft_gamlp.append(torch.zeros((labels.shape[0], y_gamlp.shape[1])))
            y_soft_gamlp[i][tr_va_te_nid] = y_gamlp
        else:
            break

    y_pred_kfold = []
    val_acc_kfold = []

    for i in range(len(y_soft_kfold)):
        y_pred_kfold.append(None)
        val_acc_kfold.append(None)

    for i in range(len(y_soft_kfold)):
        # y_soft_kfold[i] = y_soft_kfold[i].softmax(dim=-1).to(device)
        y_soft_kfold[i] = y_soft_kfold[i].to(device)
        y_pred_kfold[i] = y_soft_kfold[i].argmax(dim=-1)
        val_acc_kfold[i] = torch.sum(y_pred_kfold[i][val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        print(f'Pre valid acc: {val_acc_kfold[i]:.4f}', flush=True)

    y_pred_gamlp = []
    val_acc_gamlp = []

    for i in range(len(y_soft_gamlp)):
        y_pred_gamlp.append(None)
        val_acc_gamlp.append(None)

    print("all train", flush=True)

    for i in range(len(y_soft_gamlp)):
        y_soft_gamlp[i] = y_soft_gamlp[i].to(device)
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

    for i in range(len(y_soft_kfold)):
        y_soft_kfold[i] = y_soft_kfold[i].softmax(dim=-1).to(device)
        y_soft_kfold[i] = cs.correct(graph, y_soft_kfold[i], labels[mask_idx], mask_idx)
        y_soft_kfold[i] = cs.smooth(graph, y_soft_kfold[i], labels[mask_idx], mask_idx)
        y_pred_kfold[i] = y_soft_kfold[i].argmax(dim=-1)
        val_acc_kfold[i] = torch.sum(y_pred_kfold[i][val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        print(f'Valid acc: {val_acc_kfold[i]:.4f}', flush=True)

    print("all train", flush=True)

    for i in range(len(y_soft_gamlp)):
        y_soft_gamlp[i] = y_soft_gamlp[i].softmax(dim=-1).to(device)
        y_soft_gamlp[i] = cs.correct(graph, y_soft_gamlp[i], labels[mask_idx], mask_idx)
        y_soft_gamlp[i] = cs.smooth(graph, y_soft_gamlp[i], labels[mask_idx], mask_idx)
        y_pred_gamlp[i] = y_soft_gamlp[i].argmax(dim=-1)
        val_acc_gamlp[i] = torch.sum(y_pred_gamlp[i][val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        print(f'Valid acc: {val_acc_gamlp[i]:.4f}', flush=True)

    print(f'ensemble {len(y_soft_kfold)} models.', flush=True)
    y_soft = 0
    w = [0.2] * len(y_soft_kfold)
    # w = [
    #     0.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5,
    #     0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0
    # ]
    for i in range(len(y_soft_kfold)):
        y_soft += (w[i] * y_soft_kfold[i])

    for i in range(len(y_soft_gamlp)):
        if i != 0:
            y_soft += (w[i] * y_soft_gamlp[i])

    y_soft = y_soft.softmax(dim=-1).to(device)
    # y_soft = cs.correct(graph, y_soft, labels[mask_idx], mask_idx)
    # y_soft = cs.smooth(graph, y_soft, labels[mask_idx], mask_idx)
    y_pred = y_soft.argmax(dim=-1)
    val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    print(f'Pre valid acc: {val_acc:.4f}', flush=True)

    torch.save(y_soft[tr_va_te_nid].cpu(), '../dataset/ensem_logits.pt')
    print("save preds Done!", flush=True)

    # test
    with open(os.path.join('../dataset/test_id_dict.pkl'), 'rb') as f:
        test_id_dict = pickle.load(f)
    submit = pd.read_csv('../dataset/sample_submission_for_validation.csv')
    # submit = pd.read_csv('../dataset/sample_submission_for_test.csv')
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
    submit.to_csv(os.path.join('../outputs/', f'submit_gamlp_ensem_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

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
    parser.add_argument('--smoothing-adj', type=str, default='AD')
    parser.add_argument('--autoscale', action='store_true')
    parser.add_argument('--scale', type=float, default=1.5)
    parser.add_argument('--all_train', action='store_true')
    parser.add_argument('--num-ensemble', type=int, default=0)
    parser.add_argument('--num-seed', type=int, default=3)
    parser.add_argument('--kfold', type=int, default=8)
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
