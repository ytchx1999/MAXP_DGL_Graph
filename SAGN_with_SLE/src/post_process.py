import argparse
import glob
import os

import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

from dataset import load_dataset
from models import CorrectAndSmooth
import gc
import pickle
import pandas as pd
import time

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
    path = os.path.join(args.probs_dir, args.dataset,
                        args.model if (args.weight_style == "attention")
                        else (args.model + "_" + args.weight_style),
                        f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}" +
                        f"_K_{args.K}_label_K_{args.label_K}_probs_seed_*_stage_{args.stage}.pt")
    return path


def calculate_metrics(probs_list, labels, train_nid, val_nid, test_nid, evaluator, args, save_res=False):
    train_results = []
    val_results = []
    test_results = []
    test_nid_raw = test_nid.clone()
    inner_train_nid = torch.arange(len(train_nid))
    inner_val_nid = torch.arange(len(train_nid), len(train_nid)+len(val_nid))
    inner_test_nid = torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid))
    for probs in probs_list:
        if args.dataset in ["ppi", "yelp"]:
            preds = (probs > 0).float()
        else:
            preds = torch.argmax(probs, dim=-1)
        if evaluator != None:
            train_res = evaluator(preds[inner_train_nid], labels[train_nid])
            val_res = evaluator(preds[inner_val_nid], labels[val_nid])
            test_res = evaluator(preds[inner_test_nid], labels[test_nid])
        else:
            train_res = (preds[:len(train_nid)] == labels[train_nid]).sum().item() / len(train_nid)
            val_res = (preds[len(train_nid):(len(train_nid)+len(val_nid))] == labels[val_nid]).sum().item() / len(val_nid)
            test_res = 0.
        train_results.append(train_res)
        val_results.append(val_res)
        test_results.append(test_res)
    print(f"Train score: {np.mean(train_results):.4f}±{np.std(train_results):.4f}\n"
          f"Valid score: {np.mean(val_results):.4f}±{np.std(val_results):.4f}\n"
          f"Test score: {np.mean(test_results):.4f}±{np.std(test_results):.4f}")

    if args.dataset == 'maxp' and save_res:
        with open(os.path.join('../../dataset/test_id_dict.pkl'), 'rb') as f:
            test_id_dict = pickle.load(f)
        submit = pd.read_csv('../../dataset/sample_submission_for_validation.csv')
        with open(os.path.join('../../dataset/csv_idx_map.pkl'), 'rb') as f:
            idx_map = pickle.load(f)
        preds = torch.argmax(probs, dim=-1)
        test_seeds_list = test_nid_raw
        test_pred_list = preds[inner_test_nid]

        # save results
        for i, id in tqdm(enumerate(test_seeds_list)):
            paper_id = test_id_dict[id.item()]
            label = chr(int(test_pred_list[i].item() + 65))

            # csv_index = submit[submit['id'] == paper_id].index.tolist()[0]
            if paper_id in idx_map:
                csv_index = idx_map[paper_id]
                submit['label'][csv_index] = label

        if not os.path.exists('../../outputs'):
            os.makedirs('../../outputs', exist_ok=True)
        submit.to_csv(os.path.join('../../outputs/', f'submit_sagn_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

        print("Done!", flush=True)

    return


def main(args):
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    data = load_dataset(device, args)
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    g.ndata.pop("feat")
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    labels = labels.to(device)
    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)
    preds_path = generate_preds_path(args)
    print(preds_path)
    probs_list = load_output_files(preds_path)
    print("-"*10 + " Before " + "-"*10)
    calculate_metrics(probs_list, labels, train_nid, val_nid, test_nid, evaluator, args)

    # cs = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha, args.correction_adj,
    #                       args.num_smoothing_layers, args.smoothing_alpha, args.smoothing_adj,
    #                       autoscale=args.autoscale, scale=args.scale)
    cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                          correction_alpha=args.correction_alpha,
                          correction_adj=args.correction_adj,
                          num_smoothing_layers=args.num_smoothing_layers,
                          smoothing_alpha=args.smoothing_alpha,
                          smoothing_adj=args.smoothing_adj,
                          autoscale=args.autoscale,
                          scale=args.scale)
    processed_preds_list = []
    # inner_train_nid = torch.arange(len(train_nid))
    # inner_val_nid = torch.arange(len(train_nid), len(train_nid)+len(val_nid))
    # inner_test_nid = torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid))
    for i, probs in enumerate(probs_list):
        print(f"Processing run: {i}")
        x_all = torch.zeros((labels.shape[0], probs.shape[1]))
        x_all[tr_va_te_nid] = probs
        x_all = x_all.to(device)
        # mask_idx = train_nid.to(device)
        mask_idx = torch.cat([train_nid, val_nid]).to(device)

        y_soft_gat = torch.load('../../dataset/y_soft.pt', map_location='cpu')
        y_soft_sage = torch.load('../../dataset/y_soft_sage.pt', map_location='cpu')

        y_soft = x_all * 0.6 + y_soft_gat * 0.2 + y_soft_sage * 0.4
        y_soft = y_soft.softmax(dim=-1)
        y_soft = cs.correct(g, y_soft, labels[mask_idx], mask_idx)
        y_soft = cs.smooth(g, y_soft, labels[mask_idx], mask_idx)
        y_pred = y_soft.argmax(dim=-1)
        val_acc = torch.sum(y_pred[val_nid] == labels[val_nid]) / torch.tensor(labels[val_nid].shape[0])
        # processed_preds_list.append(cs(g, probs, labels[train_nid], args.operations, inner_train_nid, inner_val_nid, inner_test_nid, probs.size(0)))
    print("-"*10 + " Correct & Smooth " + "-"*10)
    # calculate_metrics(processed_preds_list, labels, train_nid, val_nid, test_nid, evaluator, args, save_res=True)
    print("val acc:", val_acc)

    with open(os.path.join('../../dataset/test_id_dict.pkl'), 'rb') as f:
        test_id_dict = pickle.load(f)
    submit = pd.read_csv('../../dataset/sample_submission_for_validation.csv')
    with open(os.path.join('../../dataset/csv_idx_map.pkl'), 'rb') as f:
        idx_map = pickle.load(f)
    test_pred_list = y_pred[test_nid]

    # save results
    for i, id in tqdm(enumerate(test_nid)):
        paper_id = test_id_dict[id.item()]
        label = chr(int(test_pred_list[i].item() + 65))

        # csv_index = submit[submit['id'] == paper_id].index.tolist()[0]
        if paper_id in idx_map:
            csv_index = idx_map[paper_id]
            submit['label'][csv_index] = label

    if not os.path.exists('../../outputs'):
        os.makedirs('../../outputs', exist_ok=True)
    submit.to_csv(os.path.join('../../outputs/', f'submit_sagn_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

    print("Done!", flush=True)


def define_parser():
    parser = argparse.ArgumentParser(description="hyperparameters for Correct&Smooth postprocessing")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="Select which GPU device to process (-1 for CPU)")
    parser.add_argument("--dataset", type=str, default="maxp",
                        help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="../dataset",
                        help="Root directory for datasets")
    parser.add_argument("--probs_dir", type=str, default="../intermediate_outputs",
                        help="Directory of trained model output")
    parser.add_argument("--model", type=str, default="sagn",
                        help="Model name")
    parser.add_argument("--weight_style", type=str, default="attention",
                        help="Weight style for SAGN and PlainSAGN")
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
    parser.add_argument("--num-correction-layers", type=int, default=50,
                        help="Propagation number for Correction operation")
    parser.add_argument("--num-smoothing-layers", type=int, default=50,
                        help="Propagation number for Smoothing operation")
    parser.add_argument("--correction-alpha", type=float, default=0.95,
                        help="Alpha value for Correction operation")
    parser.add_argument("--smoothing-alpha", type=float, default=0.75,
                        help="Alpha value for Smoothing operation")
    parser.add_argument("--correction-adj", type=str, default="DAD", choices=["DA", "AD", "DAD"],
                        help="Adjacency matrix for Correction operation")
    parser.add_argument("--smoothing-adj", type=str, default="DAD", choices=["DA", "AD", "DAD"],
                        help="Adjacency matrix for Smoothing operation")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Fixed scale for Correction operation (only useful when autoscale=False")
    parser.add_argument("--autoscale", action="store_true",
                        help="Whether to use autoscale in Correction operation")
    parser.add_argument("--operations", type=str, nargs="+", default=["correction", "smoothing"],
                        help="Select operations needed")
    return parser


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    main(args)
