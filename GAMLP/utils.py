import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from load_dataset import load_dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
from model import R_GAMLP, JK_GAMLP, NARS_JK_GAMLP, NARS_R_GAMLP, R_GAMLP_RLU, JK_GAMLP_RLU, NARS_JK_GAMLP_RLU, NARS_R_GAMLP_RLU


def gen_model_mag(args, num_feats, in_feats, num_classes):
    if args.method == "R_GAMLP":
        # error
        return NARS_R_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1, num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, args.pre_process, args.residual, args.pre_dropout, args.bns)
    elif args.method == "JK_GAMLP":
        return NARS_JK_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1, num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, args.pre_process, args.residual, args.pre_dropout, args.bns)


def gen_model_mag_rlu(args, num_feats, in_feats, num_classes):
    if args.method == "R_GAMLP_RLU":
        return NARS_R_GAMLP_RLU(in_feats, args.hidden, num_classes, args.num_hops+1, num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, args.pre_process, args.residual, args.pre_dropout, args.bns)
    elif args.method == "JK_GAMLP_RLU":
        return NARS_JK_GAMLP_RLU(in_feats, args.hidden, num_classes, args.num_hops+1, num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, args.pre_process, args.residual, args.pre_dropout, args.bns)


def gen_model(args, in_size, num_classes, fine_tune=None):
    if args.method == "R_GAMLP":
        return R_GAMLP(in_size, args.hidden, num_classes, args.num_hops+1,
                       args.dropout, args.input_drop, args.att_drop, args.alpha, args.n_layers_1, args.n_layers_2, args.act, args.pre_process, args.residual, args.pre_dropout, args.bns, fine_tune)
    elif args.method == "JK_GAMLP":
        return JK_GAMLP(in_size, args.hidden, num_classes, args.num_hops+1,
                        args.dropout, args.input_drop, args.att_drop, args.alpha, args.n_layers_1, args.n_layers_2, args.act, args.pre_process, args.residual, args.pre_dropout, args.bns)


def gen_model_rlu(args, in_size, num_classes):
    if args.method == "R_GAMLP_RLU":
        return R_GAMLP_RLU(in_size, args.hidden, num_classes, args.num_hops+1,
                           args.dropout, args.input_drop, args.att_drop, args.label_drop, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.pre_process, args.residual, args.pre_dropout, args.bns)

    elif args.method == "JK_GAMLP_RLU":
        return JK_GAMLP_RLU(in_size, args.hidden, num_classes, args.num_hops+1,
                            args.dropout, args.input_drop, args.att_drop, args.label_drop, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.pre_process, args.residual, args.pre_dropout, args.bns)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb, predict_prob, gama):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    iter_num = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        if label_emb != None:
            output_att = model(feat_list, label_emb[idx].to(device))
        else:
            output_att = model(feat_list)
        L1 = loss_fcn(output_att[:len(idx_1)],  y)*(len(idx_1)*1.0/(len(idx_1)+len(idx_2)))
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*(len(idx_2)*1.0/(len(idx_1)+len(idx_2)))
        loss = L1 + L3*gama
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    if evaluator != None:
        approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    else:
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        approx_acc = (y_pred == y_true).sum().item() / y_true.shape[0]
    return loss, approx_acc


def train(model, feats, labels, loss_fcn, optimizer, train_loader, label_emb, evaluator, args):
    model.train()
    device = labels.device
    total_loss = 0
    iter_num = 0
    y_true = []
    y_pred = []
    for batch in train_loader:
        optimizer.zero_grad()

        batch_feats = [x[batch].to(device) for x in feats]
        if args.flag:
            perturbs = [torch.FloatTensor(*x.shape).uniform_(-args.step_size, args.step_size).to(device) for x in batch_feats]
            perturbs = [perturb.requires_grad_() for perturb in perturbs]

            # perturbs = torch.FloatTensor(*batch_feats[0].shape).uniform_(-args.step_size, args.step_size).to(device)
            # perturbs.requires_grad_()
            # fix ...
            feat_inputs = [batch_feats[i] + perturbs[i] for i in range(len(batch_feats))]
            # feat_inputs = [batch_feats[i] + perturbs for i in range(len(batch_feats))]
        else:
            feat_inputs = batch_feats
        if label_emb != None:
            output_att = model(feat_inputs, label_emb[batch].to(device))
        else:
            output_att = model(feat_inputs)

        L1 = loss_fcn(output_att, labels[batch])
        loss_train = L1

        if args.flag:
            loss_train /= args.m
            for _ in range(args.m-1):
                loss_train.backward()
                perturb_datas = [perturb.detach() + args.step_size * torch.sign(perturb.grad.detach()) for perturb in perturbs]
                for i, perturb in enumerate(perturbs):
                    perturb.data = perturb_datas[i].data
                    perturb.grad[:] = 0

                feat_inputs = [batch_feats[i] + perturbs[i] for i in range(len(batch_feats))]

                # train_batch_logits = model(blocks, feat_input)
                if label_emb != None:
                    output_att = model(feat_inputs, label_emb[batch].to(device))
                else:
                    output_att = model(feat_inputs)
                # train_loss = cross_entropy(train_batch_logits, batch_labels)
                L1 = loss_fcn(output_att, labels[batch])
                loss_train = L1
                loss_train /= args.m

        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss = loss_train
        # optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num += 1
    loss = total_loss / iter_num
    if evaluator != None:
        acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    else:
        y_true = torch.cat(y_true, dim=0).view(-1, 1).cpu()
        y_pred = torch.cat(y_pred, dim=0)
        acc = (y_pred == y_true).sum().item() / y_true.shape[0]
    return loss, acc


@ torch.no_grad()
def test(model, feats, labels, test_loader, evaluator, label_emb):
    model.eval()
    device = labels.device
    preds = []
    true = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if label_emb != None:
            preds.append(torch.argmax(model(batch_feats, label_emb[batch].to(device)), dim=-1))
        else:
            preds.append(torch.argmax(model(batch_feats), dim=-1))
        true.append(labels[batch])
    true = torch.cat(true)
    preds = torch.cat(preds, dim=0)
    if evaluator != None:
        res = evaluator(preds, true)
    else:
        res = (preds == true).sum().item() / true.shape[0]

    return res


@ torch.no_grad()
def gen_output_torch(model, feats, test_loader, device, label_emb):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if label_emb != None:
            preds.append(model(batch_feats, label_emb[batch].to(device)).cpu())
        else:
            preds.append(model(batch_feats).cpu())
    preds = torch.cat(preds, dim=0)
    return preds
