import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
import uuid
import gc

from load_dataset import prepare_data
from utils import gen_output_torch, set_seed, train, train_rlu, test, gen_model_rlu, gen_model, gen_model_mag_rlu, gen_model_mag
import os
import pickle
import pandas as pd
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def run(args, device):
    if not os.path.exists(f'./output/{args.dataset}'):
        os.makedirs('./output/{args.dataset}')
    checkpt_file = f"./output/{args.dataset}/"+uuid.uuid4().hex

    for stage, epochs in enumerate(args.stages):
        if stage > 0 and args.use_rlu:
            predict_prob = torch.load(checkpt_file+'_{}.pt'.format(stage-1))/args.temp
            predict_prob = predict_prob.softmax(dim=1)
            train_node_nums = len(train_nid)
            valid_node_nums = len(val_nid)
            test_node_nums = len(test_nid)
            total_num_nodes = train_node_nums+valid_node_nums+test_node_nums
            if args.all_train:
                print("This history model Train ACC is {}".format(evaluator(labels[:train_node_nums], predict_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)

                print("This history model Valid ACC is {}".format(evaluator(labels[train_node_nums-valid_node_nums:train_node_nums],
                                                                            predict_prob[train_node_nums-valid_node_nums:train_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)

                print("This history model Test ACC is {}".format(evaluator(labels[train_node_nums:train_node_nums+test_node_nums],
                                                                           predict_prob[train_node_nums:train_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)
            else:
                print("This history model Train ACC is {}".format(evaluator(labels[:train_node_nums], predict_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)

                print("This history model Valid ACC is {}".format(evaluator(labels[train_node_nums:train_node_nums+valid_node_nums],
                                                                            predict_prob[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)

                print("This history model Test ACC is {}".format(evaluator(labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],
                                                                           predict_prob[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu())), flush=True)
            confident_nid = torch.arange(len(predict_prob))[
                predict_prob.max(1)[0] > args.threshold]
            extra_confident_nid = confident_nid[confident_nid >= len(
                train_nid)]
            print(f'Stage: {stage}, confident nodes: {len(extra_confident_nid)}', flush=True)
            enhance_idx = extra_confident_nid
            if len(extra_confident_nid) > 0:
                enhance_loader = torch.utils.data.DataLoader(
                    enhance_idx, batch_size=int(args.batch_size*len(enhance_idx)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)
                gc.collect()
            teacher_probs = torch.zeros(predict_prob.shape[0], predict_prob.shape[1])
            teacher_probs[enhance_idx, :] = predict_prob[enhance_idx, :]
        else:
            teacher_probs = None

        with torch.no_grad():
            data = prepare_data(device, args, teacher_probs)
        feats, labels, in_size, num_classes, \
            train_nid, val_nid, test_nid, evaluator, label_emb = data

        if args.all_train:
            train_nid = torch.cat([train_nid, val_nid], dim=0)

        if stage == 0:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=int(args.batch_size*len(train_nid)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)

        if args.all_train:
            val_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)-len(val_nid), len(train_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid), len(train_nid)+len(test_nid)), batch_size=args.batch_size,
                shuffle=False, drop_last=False)
            all_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(test_nid)), batch_size=args.batch_size,
                shuffle=False, drop_last=False)
        else:
            val_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid), len(train_nid)+len(val_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
                shuffle=False, drop_last=False)
            all_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
                shuffle=False, drop_last=False)

        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(test_nid)
        if args.all_train:
            total_num_nodes = len(train_nid) + len(test_nid)
        else:
            total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

        #num_hops = args.num_hops + 1

        if args.use_rlu == False:
            print("not use rlu", flush=True)
            if args.dataset == "ogbn-mag":
                _, num_feats, in_feats = feats[0].shape
                model = gen_model_mag(args, num_feats, in_feats, num_classes)
            else:
                model = gen_model(args, in_size, num_classes)
        else:
            print("use rlu", flush=True)
            if args.dataset == "ogbn-mag":
                _, num_feats, in_feats = feats[0].shape
                model = gen_model_mag_rlu(args, num_feats, in_feats, num_classes)
            else:
                model = gen_model_rlu(args, in_size, num_classes)
        print(model, flush=True)
        model = model.to(device)
        print("# Params:", get_n_params(model), flush=True)

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Start training
        best_epoch = 0
        best_val = 0
        best_test = 0
        count = 0

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            if stage == 0:
                loss, acc = train(model, feats, labels, loss_fcn, optimizer, train_loader, label_emb, evaluator)
            elif stage == 1:
                loss, acc = train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, feats, labels, label_emb, predict_prob, args.gama)
            else:
                loss, acc = train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, feats, labels, label_emb, predict_prob, args.gama)
            end = time.time()

            log = "Epoch {}, Time(s): {:.4f},Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, end - start, loss, acc*100)
            if epoch % args.eval_every == 0 and epoch > args.train_num_epochs[stage]:
                with torch.no_grad():
                    acc = test(model, feats, labels, val_loader, evaluator,
                               label_emb)
                end = time.time()
                log += "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
                log += "Val {:.4f}, ".format(acc)
                if acc > best_val:
                    best_epoch = epoch
                    best_val = acc

                    if args.dataset == 'maxp':
                        best_test = 0.
                    else:
                        best_test = test(model, feats, labels, test_loader, evaluator,
                                         label_emb)
                    torch.save(model.state_dict(), checkpt_file+f'_{stage}.pkl')
                    count = 0
                else:
                    count = count+args.eval_every
                    if count >= args.patience:
                        break
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(
                    best_epoch, best_val, best_test)
            print(log, flush=True)

        print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            best_epoch, best_val, best_test), flush=True)

        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        preds = gen_output_torch(model, feats, all_loader, labels.device, label_emb)
        torch.save(preds, checkpt_file+f'_{stage}.pt')
        # torch.save(preds, f'./output/{args.dataset}/gamlp_{stage}.pt')

    torch.save(preds, '../dataset/gamlp.pt')

    if args.dataset == 'maxp':
        with open(os.path.join('../dataset/test_id_dict.pkl'), 'rb') as f:
            test_id_dict = pickle.load(f)
        submit = pd.read_csv('../dataset/sample_submission_for_validation.csv')
        with open(os.path.join('../dataset/csv_idx_map.pkl'), 'rb') as f:
            idx_map = pickle.load(f)
        preds = torch.argmax(preds, dim=-1)
        test_seeds_list = test_nid
        if args.all_train:
            test_pred_list = preds[len(train_nid):len(train_nid)+len(test_nid)]
        else:
            test_pred_list = preds[len(train_nid)+len(val_nid):len(train_nid)+len(val_nid)+len(test_nid)]

        # save results
        for i, id in tqdm(enumerate(test_seeds_list)):
            paper_id = test_id_dict[id.item()]
            label = chr(int(test_pred_list[i].item() + 65))

            # csv_index = submit[submit['id'] == paper_id].index.tolist()[0]
            if paper_id in idx_map:
                csv_index = idx_map[paper_id]
                submit['label'][csv_index] = label

        if not os.path.exists('../outputs'):
            os.makedirs('../outputs', exist_ok=True)
        submit.to_csv(os.path.join('../outputs/', f'submit_gamlp_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

        print("Done!", flush=True)

    return best_val, best_test, preds


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training", flush=True)
        set_seed(args.seed+i)
        best_val, best_test, preds = run(args, device)
        np.save(f"output/{args.dataset}/output_{i}.npy", preds.numpy())
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
          f"std: {np.std(val_accs):.4f}", flush=True)
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
          f"std: {np.std(test_accs):.4f}", flush=True)

    return np.mean(test_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMLP")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--label-num-hops", type=int, default=9,
                        help="number of hops for label")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed used in the training")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="maxp")  # ogbn-products
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--n-layers-1", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="initial residual parameter for the model")
    parser.add_argument("--temp", type=float, default=1,
                        help="temperature of the output prediction")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="the threshold for the node to be added into the model")
    parser.add_argument("--input-drop", type=float, default=0,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5,
                        help="label feature dropout of model")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--method", type=str, default="JK_GAMLP",
                        help="the model to use")
    parser.add_argument("--use-emb", type=str)
    parser.add_argument("--root", type=str, default='../dataset')
    parser.add_argument("--emb_path", type=str, default='/data4/zwt/NARS-main')
    parser.add_argument("--use-relation-subsets", type=str, default='/data4/zwt/NARS-main/sample_relation_subsets/examples/mag')
    parser.add_argument("--use-rlu", action='store_true', default=False,
                        help="whether to use the reliable data distillation")
    parser.add_argument("--train-num-epochs", nargs='+', type=int, default=[100, 100],
                        help="The Train epoch setting for each stage.")
    parser.add_argument("--stages", nargs='+', type=int, default=[300, 300],
                        help="The epoch setting for each stage.")
    parser.add_argument("--pre-dropout", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--all-train", action="store_true")

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
