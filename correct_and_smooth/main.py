import argparse
import copy
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model import MLP, MLPLinear, CorrectAndSmooth
from utils import load_dgl_graph, time_diff
import pandas as pd
import pickle
from tqdm import tqdm
import time


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

    # y_soft = model(feats).exp()

    # y_soft = torch.rand(labels.shape[0], 23)
    y_soft = torch.load('../dataset/y_soft.pt', map_location='cpu')
    # use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt

    y_soft_sage = None
    y_soft_conv = None
    if os.path.exists('../dataset/y_soft_sage.pt'):
        y_soft_sage = torch.load('../dataset/y_soft_sage.pt', map_location='cpu')
    if os.path.exists('../dataset/y_soft_sage.pt'):
        y_soft_conv = torch.load('../dataset/y_soft_conv.pt', map_location='cpu')
    # if y_soft_sage != None and y_soft_conv != None:
    #     y_soft = 0.6 * y_soft + 0.2 * y_soft_sage + 0.2 * y_soft_conv

    y_soft = y_soft.softmax(dim=-1).to(device)
    y_soft_sage = y_soft_sage.softmax(dim=-1).to(device)
    y_soft_conv = y_soft_conv.softmax(dim=-1).to(device)

    y_pred = y_soft.argmax(dim=-1)
    y_pred_sage = y_soft_sage.argmax(dim=-1)
    y_pred_conv = y_soft_conv.argmax(dim=-1)

    val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    val_acc_sage = torch.sum(y_pred_sage[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    val_acc_conv = torch.sum(y_pred_conv[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])

    print(f'Pre valid acc: {val_acc:.4f}', flush=True)
    print(f'Pre valid acc: {val_acc_sage:.4f}', flush=True)
    print(f'Pre valid acc: {val_acc_conv:.4f}', flush=True)

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
    y_soft = cs.correct(graph, y_soft, labels[mask_idx], mask_idx)
    y_soft = cs.smooth(graph, y_soft, labels[mask_idx], mask_idx)
    y_pred = y_soft.argmax(dim=-1)
    val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])

    y_soft_sage = cs.correct(graph, y_soft_sage, labels[mask_idx], mask_idx)
    y_soft_sage = cs.smooth(graph, y_soft_sage, labels[mask_idx], mask_idx)
    y_pred_sage = y_soft_sage.argmax(dim=-1)
    val_acc_sage = torch.sum(y_pred_sage[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])

    y_soft_conv = cs.correct(graph, y_soft_conv, labels[mask_idx], mask_idx)
    y_soft_conv = cs.smooth(graph, y_soft_conv, labels[mask_idx], mask_idx)
    y_pred_conv = y_soft_conv.argmax(dim=-1)
    val_acc_conv = torch.sum(y_pred_conv[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])

    print(f'Valid acc: {val_acc:.4f}', flush=True)
    print(f'Valid acc: {val_acc_sage:.4f}', flush=True)
    print(f'Valid acc: {val_acc_conv:.4f}', flush=True)

    y_soft = 0.5 * y_soft + 0.4 * y_soft_sage + 0.1 * y_soft_conv
    y_pred = y_soft.argmax(dim=-1)
    val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
    print(f'Final valid acc: {val_acc:.4f}', flush=True)

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
    submit.to_csv(os.path.join('../outputs/', f'submit_cs_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

    print("Done!", flush=True)


if __name__ == '__main__':
    """
    Correct & Smoothing Hyperparameters
    """
    parser = argparse.ArgumentParser(description='Base predictor(C&S)')

    # Dataset
    parser.add_argument('--gpu', type=int, default=2, help='-1 for cpu')
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

    args = parser.parse_args()
    print(args, flush=True)

    main()
