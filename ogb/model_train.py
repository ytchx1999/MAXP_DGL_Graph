# -*- coding:utf-8 -*-

# Author:james Zhang
"""
    Minibatch training with node neighbor sampling in multiple GPUs
"""

import os
import argparse
import datetime as dt
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from models import GraphSageModel, GraphConvModel, GraphAttnModel
from utils import load_dgl_graph, load_dgl_ogb_graph, time_diff
from model_utils import early_stopper, thread_wrapped_func
import pickle
import pandas as pd
import time
from tqdm import tqdm
import math


epsilon = 1 - math.log(2)


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def load_subtensor(node_feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def cleanup():
    dist.destroy_process_group()


def cpu_train(graph_data,
              gnn_model,
              hidden_dim,
              n_layers,
              n_classes,
              fanouts,
              batch_size,
              device,
              num_workers,
              epochs,
              out_path):
    """
        运行在CPU设备上的训练代码。由于数据量很大，仅仅用于代码调试。建议有GPU的，请使用下面的GPU设备训练的代码已提高训练速度。
    """
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers)

    # 2 initialize GNN model
    in_feat = node_feat.shape[1]

    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([5] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device)

    # 3 define loss function and optimizer
    loss_fn = thnn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 4 train epoch
    avg = 0
    iter_tput = []
    start_t = dt.datetime.now()

    print('Start training at: {}-{} {}:{}:{}'.format(start_t.month,
                                                     start_t.day,
                                                     start_t.hour,
                                                     start_t.minute,
                                                     start_t.second), flush=True)

    for epoch in range(epochs):

        for step, (input_nodes, seeds, mfgs) in enumerate(train_dataloader):

            start_t = dt.datetime.now()

            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device)
            mfgs = [mfg.to(device) for mfg in mfgs]

            batch_logit = model(mfgs, batch_inputs)
            loss = loss_fn(batch_logit, batch_labels)
            pred = th.sum(th.argmax(batch_logit, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_t1 = dt.datetime.now()
            h, m, s = time_diff(e_t1, start_t)

            print('In epoch:{:03d}|batch:{}, loss:{:4f}, acc:{:4f}, time:{}h{}m{}s'.format(epoch,
                                                                                           step,
                                                                                           loss,
                                                                                           pred.detach(),
                                                                                           h, m, s), flush=True)

    # 5 save model if need
    #     pass


def gpu_train(proc_id, n_gpus, GPUS,
              graph_data, gnn_model,
              hidden_dim, n_layers, n_classes, fanouts, test_fanouts, args,
              batch_size=32, num_workers=4, epochs=100, message_queue=None,
              output_folder='./output'):

    device_id = GPUS[proc_id]
    print('Use GPU {} for training ......'.format(device_id), flush=True)

    # ------------------- 1. Prepare data and split for multiple GPUs ------------------- #
    start_t = dt.datetime.now()
    print('Start graph building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second), flush=True)

    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    train_div, _ = divmod(train_nid.shape[0], n_gpus)
    val_div, _ = divmod(val_nid.shape[0], n_gpus)
    test_div, _ = divmod(test_nid.shape[0], n_gpus)

    # just use one GPU, give all training/validation index to the one GPU
    if proc_id == (n_gpus - 1):
        train_nid_per_gpu = train_nid[proc_id * train_div:]
        val_nid_per_gpu = val_nid[proc_id * val_div:]
        test_nid_per_gpu = test_nid[proc_id * test_div:]
        # use valid and test
        if args.all_train:
            train_nid_per_gpu = np.concatenate((train_nid_per_gpu, val_nid_per_gpu, test_nid_per_gpu), axis=0)
    # in case of multiple GPUs, split training/validation index to different GPUs
    else:
        train_nid_per_gpu = train_nid[proc_id * train_div: (proc_id + 1) * train_div]
        val_nid_per_gpu = val_nid[proc_id * val_div: (proc_id + 1) * val_div]
        test_nid_per_gpu = test_nid[proc_id * test_div: (proc_id + 1) * test_div]
        # use valid
        if args.all_train:
            train_nid_per_gpu = np.concatenate((train_nid_per_gpu, val_nid_per_gpu, test_nid_per_gpu), axis=0)

    sampler = MultiLayerNeighborSampler(fanouts)
    test_sampler = MultiLayerNeighborSampler(test_fanouts)  # test
    # test_sampler = MultiLayerFullNeighborSampler(len(test_fanouts))
    train_dataloader = NodeDataLoader(graph,
                                      train_nid_per_gpu,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      )
    val_dataloader = NodeDataLoader(graph,
                                    val_nid_per_gpu,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )
    test_dataloader = NodeDataLoader(graph,
                                     test_nid_per_gpu,
                                     test_sampler,
                                     batch_size=64,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=num_workers,
                                     )
    # if args.all_train:
    #     all_nid_per_gpu = np.concatenate((train_nid_per_gpu, test_nid_per_gpu), axis=0)
    #     # all_nid_per_gpu = train_nid_per_gpu
    # else:
    #     all_nid_per_gpu = np.concatenate((train_nid_per_gpu, val_nid_per_gpu, test_nid_per_gpu), axis=0)
    # all_sampler = MultiLayerNeighborSampler(test_fanouts)
    # graph_loader = NodeDataLoader(graph,
    #                               all_nid_per_gpu,
    #                               all_sampler,
    #                               batch_size=256,
    #                               shuffle=False,
    #                               drop_last=False,
    #                               num_workers=num_workers,
    #                               )
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s), flush=True)

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    start_t = dt.datetime.now()
    print('Start Model building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second), flush=True)

    if n_gpus > 1:
        dist_init_method = 'tcp://{}:{}'.format('127.0.0.1', '23456')
        world_size = n_gpus
        dist.init_process_group(backend='nccl',
                                init_method=dist_init_method,
                                world_size=world_size,
                                rank=proc_id)

    in_feat = node_feat.shape[1]
    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([4] * n_layers), activation=F.relu, feat_drop=0.2, attn_drop=0.1)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device_id)

    if n_gpus > 1:
        model = thnn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device_id],
                                                      output_device=device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s), flush=True)

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # lr adjustment
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8,
                                                        patience=1000, verbose=True)

    earlystoper = early_stopper(patience=2, verbose=False)

    # ------------------- 4. Train model  ----------------------------------------------- #
    print('Plan to train {} epoches \n'.format(epochs), flush=True)

    for epoch in range(epochs):
        # mini-batch for training
        train_loss_list = []
        # train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]

            if args.flag:
                perturb = th.FloatTensor(*batch_inputs.shape).uniform_(-args.step_size, args.step_size).to(device_id)
                perturb.requires_grad_()
                feat_input = batch_inputs + perturb
            else:
                feat_input = batch_inputs
            # metric and loss
            train_batch_logits = model(blocks, feat_input)
            # train_loss = loss_fn(train_batch_logits, batch_labels)
            train_loss = cross_entropy(train_batch_logits, batch_labels)

            if args.flag:
                train_loss /= args.m
                for _ in range(args.m-1):
                    train_loss.backward()
                    perturb_data = perturb.detach() + args.step_size * th.sign(perturb.grad.detach())
                    perturb.data = perturb_data.data
                    perturb.grad[:] = 0

                    feat_input = batch_inputs + perturb

                    train_batch_logits = model(blocks, feat_input)
                    train_loss = cross_entropy(train_batch_logits, batch_labels)
                    train_loss /= args.m
            # backward
            # optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.cpu().detach().numpy())
            tr_batch_pred = th.sum(th.argmax(train_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            scheduler.step(train_loss)

            if step % 10 == 0:
                print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(train_loss_list),
                                                                                                tr_batch_pred.detach()), flush=True)

        # mini-batch for validation
        # best_val_acc = 0

        if not args.all_train:
            print("Validation...", flush=True)
            val_loss_list = []
            val_acc_list = []
            model.eval()
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                # forward
                batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
                blocks = [block.to(device_id) for block in blocks]
                # metric and loss
                val_batch_logits = model(blocks, batch_inputs)
                # val_loss = loss_fn(val_batch_logits, batch_labels)
                val_loss = cross_entropy(val_batch_logits, batch_labels)

                val_loss_list.append(val_loss.detach().cpu().numpy())
                val_batch_pred = th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

                if step % 10 == 0:
                    print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(val_loss_list),
                                                                                                val_batch_pred.detach()), flush=True)
            # put validation results into message queue and aggregate at device 0
            if n_gpus > 1 and message_queue != None:
                message_queue.put(val_loss_list)

                if proc_id == 0:
                    for i in range(n_gpus):
                        loss = message_queue.get()
                        print(loss, flush=True)
                        del loss
            else:
                print(val_loss_list, flush=True)

            print("Test...", flush=True)
            test_loss_list = []
            test_acc_list = []
            model.eval()
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # forward
                batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
                blocks = [block.to(device_id) for block in blocks]
                # metric and loss
                test_batch_logits = model(blocks, batch_inputs)
                # val_loss = loss_fn(val_batch_logits, batch_labels)
                test_loss = cross_entropy(test_batch_logits, batch_labels)

                test_loss_list.append(test_loss.detach().cpu().numpy())
                test_batch_pred = th.sum(th.argmax(test_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

                if step % 10 == 0:
                    print('In epoch:{:03d}|batch:{:04d}, test_loss:{:4f}, test_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(test_loss_list),
                                                                                                test_batch_pred.detach()), flush=True)
            # put validation results into message queue and aggregate at device 0
            if n_gpus > 1 and message_queue != None:
                message_queue.put(test_loss_list)

                if proc_id == 0:
                    for i in range(n_gpus):
                        loss = message_queue.get()
                        print(loss, flush=True)
                        del loss
            else:
                print(test_loss_list, flush=True)

    # test
    # if not args.save_emb:
    #     with open(os.path.join('../dataset/test_id_dict.pkl'), 'rb') as f:
    #         test_id_dict = pickle.load(f)
    #     submit = pd.read_csv('../dataset/sample_submission_for_validation.csv')
    #     with open(os.path.join('../dataset/csv_idx_map.pkl'), 'rb') as f:
    #         idx_map = pickle.load(f)

    #     test_seeds_list = []
    #     test_pred_list = []
    #     model.eval()
    #     for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
    #         # print('test_batch:', step)
    #         # forward
    #         batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
    #         blocks = [block.to(device_id) for block in blocks]
    #         # metric and loss
    #         test_batch_logits = model(blocks, batch_inputs)
    #         # test_loss = loss_fn(test_batch_logits, batch_labels`)

    #         test_pred = th.argmax(test_batch_logits, dim=1)

    #         test_seeds_list.append(seeds)
    #         test_pred_list.append(test_pred)

    #         if step % 10 == 0:
    #             print('test batch:{:04d}'.format(step), flush=True)

    #     test_seeds_list = th.cat(test_seeds_list, dim=0)
    #     test_pred_list = th.cat(test_pred_list, dim=0)

    #     # save results
    #     for i, id in tqdm(enumerate(test_seeds_list)):
    #         paper_id = test_id_dict[id.item()]
    #         label = chr(int(test_pred_list[i].item() + 65))

    #         # csv_index = submit[submit['id'] == paper_id].index.tolist()[0]
    #         if paper_id in idx_map:
    #             csv_index = idx_map[paper_id]
    #             submit['label'][csv_index] = label

    #     if not os.path.exists('../outputs'):
    #         os.makedirs('../outputs', exist_ok=True)
    #     submit.to_csv(os.path.join('../outputs/', f'submit_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

    # # save emb for lp and cs
    # if args.save_emb:
    #     print("Saving inference emb...", flush=True)
    #     model.eval()

    #     node_num = labels.shape[0]
    #     x_all = th.zeros(node_num, 23).cpu()
    #     for step, (input_nodes, seeds, blocks) in enumerate(graph_loader):
    #         # print('test_batch:', step)
    #         # forward
    #         batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
    #         blocks = [block.to(device_id) for block in blocks]
    #         # metric and loss
    #         batch_logits = model(blocks, batch_inputs)
    #         # test_loss = loss_fn(test_batch_logits, batch_labels`)
    #         xs = batch_logits.detach().data.cpu()
    #         x_all[seeds] += xs
    #         # collect.append(xs)

    #         # test_pred = th.argmax(batch_logits, dim=1)

    #         if step % 10 == 0:
    #             print('inference batch:{:04d}'.format(step), flush=True)

    #     # x_all = th.cat(xs, dim=0)
    #     print(x_all.shape, flush=True)
    #     if args.gnn_model == 'graphattn':
    #         th.save(x_all, '../dataset/y_soft.pt')
    #     elif args.gnn_model == 'graphsage':
    #         th.save(x_all, '../dataset/y_soft_sage.pt')
    #     elif args.gnn_model == 'graphconv':
    #         th.save(x_all, '../dataset/y_soft_conv.pt')

    # -------------------------5. Collect stats ------------------------------------#
    # best_preds = earlystoper.val_preds
    # best_logits = earlystoper.val_logits
    #
    # best_precision, best_recall, best_f1 = get_f1_score(val_y.cpu().numpy(), best_preds)
    # best_auc = get_auc_score(val_y.cpu().numpy(), best_logits[:, 1])
    # best_recall_at_99precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.99)
    # best_recall_at_90precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.9)

    # plot_roc(val_y.cpu().numpy(), best_logits[:, 1])
    # plot_p_r_curve(val_y.cpu().numpy(), best_logits[:, 1])

    # -------------------------6. Save models --------------------------------------#
    model_path = os.path.join(output_folder, 'dgl_ogb_model-' + '{:06d}'.format(np.random.randint(100000)) + '.pth')

    if n_gpus > 1:
        if proc_id == 0:
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)
            # after trainning, remember to cleanup and release resouces
            cleanup()
    else:
        model_para_dict = model.state_dict()
        th.save(model_para_dict, model_path)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
    parser.add_argument('--data_path', type=str, default='../dataset')
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn'], default='graphattn')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument("--fanout", type=str, default='15,15,15')
    parser.add_argument("--test_fanout", type=str, default='100,100,100')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--GPU', nargs='+', type=int, default=0)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--out_path', type=str, default='../outputs')
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--all_train', action="store_true")
    parser.add_argument('--use_label', action="store_true")
    parser.add_argument('--use_emb', action="store_true")
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--flag', action="store_true")
    parser.add_argument('--ogb', action="store_true")
    args = parser.parse_args()

    # parse arguments
    BASE_PATH = args.data_path
    MODEL_CHOICE = args.gnn_model
    HID_DIM = args.hidden_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    TEST_FANOUTS = [int(i) for i in args.test_fanout.split(',')]
    BATCH_SIZE = args.batch_size
    GPUS = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs
    OUT_PATH = args.out_path

    # output arguments for logging
    print('Data path: {}'.format(BASE_PATH), flush=True)
    print('Used algorithm: {}'.format(MODEL_CHOICE), flush=True)
    print('Hidden dimensions: {}'.format(HID_DIM), flush=True)
    print('number of hidden layers: {}'.format(N_LAYERS), flush=True)
    print('Fanout list: {}'.format(FANOUTS), flush=True)
    print('Batch size: {}'.format(BATCH_SIZE), flush=True)
    print('GPU list: {}'.format(GPUS), flush=True)
    print('Number of workers per GPU: {}'.format(WORKERS), flush=True)
    print('Max number of epochs: {}'.format(EPOCHS), flush=True)
    print('Output path: {}'.format(OUT_PATH), flush=True)

    # Retrieve preprocessed data and add reverse edge and self-loop
    if args.ogb:
        graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_ogb_graph(BASE_PATH)
    else:
        graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    if args.ogb:
        num_classes = 172
    else:
        num_classes = 23

    # if args.use_emb:
    #     print("Use node2vec embedding...", flush=True)
    #     emb = th.load('../dataset/emb.pt', map_location='cpu')
    #     emb.requires_grad = False
    #     node_feat = th.cat([node_feat, emb], dim=1)

    # # add labels
    # if args.use_label:
    #     print("Use labels...", flush=True)
    #     onehot = th.zeros(labels.shape[0], 23)
    #     if args.all_train:
    #         onehot[np.concatenate((train_nid, val_nid), axis=0), labels[np.concatenate((train_nid, val_nid), axis=0)]] = 1
    #     else:
    #         onehot[train_nid, labels[train_nid]] = 1
    #     node_feat = th.cat([node_feat, onehot], dim=1)

    # call train with CPU, one GPU, or multiple GPUs
    if GPUS[0] < 0:
        cpu_device = th.device('cpu')
        cpu_train(graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                  gnn_model=MODEL_CHOICE,
                  n_layers=N_LAYERS,
                  hidden_dim=HID_DIM,
                  n_classes=num_classes,
                  fanouts=FANOUTS,
                  batch_size=BATCH_SIZE,
                  num_workers=WORKERS,
                  device=cpu_device,
                  epochs=EPOCHS,
                  out_path=OUT_PATH)
    else:
        n_gpus = len(GPUS)

        if n_gpus == 1:
            gpu_train(0, n_gpus, GPUS,
                      graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                      gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=num_classes,
                      fanouts=FANOUTS, test_fanouts=TEST_FANOUTS, args=args, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS,
                      message_queue=None, output_folder=OUT_PATH)
        else:
            message_queue = mp.Queue()
            procs = []
            for proc_id in range(n_gpus):
                p = mp.Process(target=gpu_train,
                               args=(proc_id, n_gpus, GPUS,
                                     (graph, labels, train_nid, val_nid, test_nid, node_feat),
                                     MODEL_CHOICE, HID_DIM, N_LAYERS, num_classes,
                                     FANOUTS, BATCH_SIZE, WORKERS, EPOCHS,
                                     message_queue, OUT_PATH))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
