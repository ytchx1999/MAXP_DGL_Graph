# MAXP_DGL_Graph
[ 2021 MAXP 命题赛 任务一：[基于DGL的图机器学习任务 ](https://www.biendata.xyz/competition/maxp_dgl/) ] | [ Team: Graph@ICT ]



## 代码库包括2个部分：


1. 用于数据预处理的4个Jupyter Notebook
2. 用DGL构建的3个GNN模型(GCN,GraphSage和GAT)，以及训练模型所用的代码和辅助函数。

## 依赖包：

```bash
dgl==0.7.1
pytorch==1.7.0
sklearn
pandas
numpy
datetime
tqdm
```

## 环境安装
```bash
pip install -r requirement.txt
```

## 如何运行：

### 运行jupyter进行数据预处理
对于4个Jupyter Notebook文件，请使用Jupyter环境运行，并注意把其中的竞赛数据文件所在的文件夹替换为你自己保存数据文件的文件夹。
并记录下你处理完成后的数据文件所在的位置，供下面模型训练使用。

### 运行node2vec并保存Embedding
```bash
cd node2vec/
# run node2vec in backward
nohup python main.py > ../outputs/node2vec.log 2>&1 &
tail -f ../outputs/node2vec.log
```
结果保存在`../dataset/emb.pt`中。

## SAGN模型
进入SAGN_with_SLE文件夹，按照指示进行运行。


### 运行GNN模型并保存test结果
对于GNN的模型，需要先cd到gnn目录，然后运行：

```bash
cd gnn/
# generate index map
python csv_idx_map.py
# then run gnn in backward
nohup python3 model_train.py --GPU 1 --use_emb --use_label --flag --all_train > ../outputs/train1.log 2>&1 &
# check the result in terminal
tail -f ../outputs/train1.log

# or
python3 model_train.py --data_path ../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 1 --out_path ./
```
test结果保存在`../outputs/submit_xxxx-xx-xx.csv`中。

### 运行和使用C&S方法

```bash
# pretrain model in backward
cd gnn/
# graphattn
nohup python3 model_train.py --GPU 1 --use_emb --save_emb --flag --all_train > ../outputs/train1.log 2>&1 &
# graphsage
nohup python3 model_train.py --GPU 0 --use_emb --save_emb --gnn_model graphsage --flag --all_train > ../outputs/train.log 2>&1 &
# graphconv
nohup python3 model_train.py --GPU 1 --use_emb --save_emb --gnn_model graphconv --flag --all_train > ../outputs/train1.log 2>&1 &
# run c&s
cd ../correct_and_smooth
python3 main.py --all_train
```
test结果保存在`../outputs/submit_cs_xxxx-xx-xx.csv`中。

*注意*：请把--data_path的路径替换成用Jupyter Notebook文件处理后数据所在的位置路径。其余的参数，请参考model_train.py里面的入参说明修改。

如果希望使用单GPU进行模型训练，则需要修改入参 `--GPU`的输入值为单个GPU的编号，如：

```bash
--GPU 0
```

如果希望使用单机多GPU进行模型训练，则需要修改入参 `--GPU`的输入值为多个可用的GPU的编号，并用空格分割，如：

```bash
--GPU 0 1 2 3
```

## ogbn-papers100M预训练
ogbn-papers100M进行训练并保存model参数。
```bash
cd ogb/
# then run gnn in backward
nohup python3 model_train.py --GPU 0 --ogb --all_train > ../outputs/ogb.log 2>&1 &
# run model in this dataset
cd ../gnn/
nohup python3 model_train.py --GPU 1 --pretrain --use_emb --save_emb --all_train > ../outputs/train1.log 2>&1 &
```

---
## 其他说明
### 如何调试
进入gnn目录，配置`launch.json`，如下面所示（解析参数），点击开始调试。
```javascript
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 当前文件",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}",
            "justMyCode": false,
            // python model_train.py --data_path ../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 1 --out_path ./
            "args": [
                "--data_path",
                "../dataset",
                "--gnn_model",
                "graphsage",
                "--hidden_dim",
                "64",
                "--n_layers",
                "2",
                "--fanout",
                "20,20",
                "--batch_size",
                "4096",
                "--GPU",
                "1",
                "--out_path",
                "./outputs"
            ]
        }
    ]
}
```

### Tree
```bash
.
├── correct_and_smooth
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── __pycache__
│   │   ├── model.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   └── utils.py
├── dataset
│   ├── csv_idx_map.pkl
│   ├── diff_nodes.csv
│   ├── emb.pt
│   ├── features.npy
│   ├── graph.bin
│   ├── IDandLabels.csv
│   ├── labels.pkl
│   ├── link_phase1.csv
│   ├── sample_submission_for_validation.csv
│   ├── sgc_emb.pt
│   ├── test_id_dict.pkl
│   ├── train_nodes.csv
│   ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
│   ├── validation_nodes.csv
│   ├── y_soft_conv.pt
│   ├── y_soft.pt
│   └── y_soft_sage.pt
├── gnn
│   ├── csv_idx_map.py
│   ├── flag.py
│   ├── __init__.py
│   ├── models.py
│   ├── model_train.py
│   ├── model_utils.py
│   ├── __pycache__
│   │   ├── models.cpython-37.pyc
│   │   ├── model_utils.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── test_chx.py
│   ├── test.ipynb
│   └── utils.py
├── LICENSE
├── MAXP 2021初赛数据探索和处理-1.ipynb
├── MAXP 2021初赛数据探索和处理-2.ipynb
├── MAXP 2021初赛数据探索和处理-3.ipynb
├── MAXP 2021初赛数据探索和处理-4.ipynb
├── node2vec
│   ├── main.py
│   ├── model.py
│   ├── __pycache__
│   │   ├── model.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   └── utils.py
├── ogb
│   ├── csv_idx_map.py
│   ├── flag.py
│   ├── __init__.py
│   ├── models.py
│   ├── model_train.py
│   ├── model_utils.py
│   ├── __pycache__
│   │   ├── models.cpython-37.pyc
│   │   ├── model_utils.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── test_chx.py
│   ├── test.ipynb
│   └── utils.py
├── ogbn_papers100M
│   ├── mapping
│   │   ├── labelidx2arxivcategeory.csv.gz
│   │   ├── nodeidx2paperid.csv.gz
│   │   └── README.md
│   ├── processed
│   │   └── dgl_data_processed
│   ├── raw
│   │   ├── data.npz
│   │   └── node-label.npz
│   ├── RELEASE_v1.txt
│   └── split
│       └── time
│           ├── test.csv.gz
│           ├── train.csv.gz
│           └── valid.csv.gz
├── outputs
│   ├── dgl_ogb_model-081260.pth
│   ├── dgl_ogb_model_gat.pth
│   ├── node2vec.log
│   ├── ogb1.log
│   ├── ogb.log
│   ├── submit_2021-10-13.csv
│   ├── submit_2021-10-14.csv
│   ├── submit_2021-10-15.csv
│   ├── submit_2021-10-17.csv
│   ├── submit_2021-10-18.csv
│   ├── submit_2021-10-19.csv
│   ├── submit_2021-10-20.csv
│   ├── submit_cs_2021-10-20.csv
│   ├── submit_cs_2021-10-21.csv
│   ├── submit_cs_2021-10-22.csv
│   ├── submit_cs_2021-10-23.csv
│   ├── submit_cs_2021-10-24.csv
│   ├── submit_cs_2021-10-25.csv
│   ├── submit_cs_2021-10-26.csv
│   ├── submit_cs_2021-10-29.csv
│   ├── submit_sagn_2021-10-30.csv
│   ├── submit_sagn_2021-10-31.csv
│   ├── train1.log
│   └── train.log
├── README 2.md
├── README.md
├── requirements.txt
├── SAGN_with_SLE
│   ├── converge_stats
│   │   └── ogbn-products
│   │       └── sagn_uniform
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_0.csv
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_0.png
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_1.csv
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_1.png
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_2.csv
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_2.png
│   │           ├── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_0.csv
│   │           ├── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_0.png
│   │           ├── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_1.csv
│   │           ├── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_1.png
│   │           ├── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_2.csv
│   │           └── val_loss_use_labels_True_use_feats_True_K_5_label_K_9_seed_0_stage_2.png
│   ├── dataset
│   │   ├── maxp
│   │   │   ├── embedding
│   │   │   │   ├── smoothed_embs_K_5.pt
│   │   │   │   └── smoothed_label_emb_K_9.pt
│   │   │   └── tmp
│   │   ├── ogbn_products
│   │   │   ├── embedding
│   │   │   ├── mapping
│   │   │   │   ├── labelidx2productcategory.csv.gz
│   │   │   │   ├── nodeidx2asin.csv.gz
│   │   │   │   └── README.md
│   │   │   ├── processed
│   │   │   │   ├── dgl_data_processed
│   │   │   │   ├── geometric_data_processed.pt
│   │   │   │   ├── pre_filter.pt
│   │   │   │   └── pre_transform.pt
│   │   │   ├── raw
│   │   │   │   ├── edge.csv.gz
│   │   │   │   ├── node-feat.csv.gz
│   │   │   │   ├── node-label.csv.gz
│   │   │   │   ├── num-edge-list.csv.gz
│   │   │   │   └── num-node-list.csv.gz
│   │   │   ├── RELEASE_v1.txt
│   │   │   ├── split
│   │   │   │   └── sales_ranking
│   │   │   │       ├── test.csv.gz
│   │   │   │       ├── train.csv.gz
│   │   │   │       └── valid.csv.gz
│   │   │   └── tmp
│   │   └── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
│   ├── intermediate_outputs
│   │   ├── maxp
│   │   │   └── sagn
│   │   │       ├── use_labels_False_use_feats_True_K_3_label_K_9_probs_seed_0_stage_0.pt
│   │   │       ├── use_labels_False_use_feats_True_K_3_label_K_9_probs_seed_0_stage_1.pt
│   │   │       ├── use_labels_False_use_feats_True_K_3_label_K_9_probs_seed_0_stage_2.pt
│   │   │       ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_0.pt
│   │   │       ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_1.pt
│   │   │       ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
│   │   │       ├── use_labels_True_use_feats_True_K_3_label_K_9_probs_seed_0_stage_0.pt
│   │   │       ├── use_labels_True_use_feats_True_K_3_label_K_9_probs_seed_0_stage_1.pt
│   │   │       ├── use_labels_True_use_feats_True_K_3_label_K_9_probs_seed_0_stage_2.pt
│   │   │       └── use_labels_True_use_feats_True_K_3_label_K_9_probs_seed_0_stage_3.pt
│   │   └── ogbn-products
│   │       └── sagn_uniform
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_probs_seed_0_stage_0.pt
│   │           ├── use_labels_True_use_feats_True_K_5_label_K_9_probs_seed_0_stage_1.pt
│   │           └── use_labels_True_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
│   ├── LICENSE
│   ├── outputs
│   │   └── sagn.log
│   ├── README.md
│   ├── scripts
│   │   ├── cora
│   │   │   └── train_cora_sagn_use_labels.sh
│   │   ├── flickr
│   │   │   └── train_flickr_sagn.sh
│   │   ├── maxp
│   │   │   └── train_maxp_sagn_use_label.sh
│   │   ├── ogbn-mag
│   │   │   ├── train_ogbn_mag_sagn.sh
│   │   │   └── train_ogbn_mag_sagn_TransE.sh
│   │   ├── ogbn-papers100M
│   │   │   ├── train_ogbn_papers100M_sagn_no_labels.sh
│   │   │   └── train_ogbn_papers100M_sagn_use_label.sh
│   │   ├── ogbn-products
│   │   │   ├── postprocess_ogbn_products_sagn_use_labels.sh
│   │   │   ├── train_ogbn_products_mlp_no_labels.sh
│   │   │   ├── train_ogbn_products_mlp_use_labels.sh
│   │   │   ├── train_ogbn_products_sagn_no_features.sh
│   │   │   ├── train_ogbn_products_sagn_no_labels.sh
│   │   │   ├── train_ogbn_products_sagn_use_labels_morestages.sh
│   │   │   ├── train_ogbn_products_sagn_use_labels.sh
│   │   │   ├── train_ogbn_products_sign_no_labels.sh
│   │   │   ├── train_ogbn_products_sign_use_labels.sh
│   │   │   ├── train_ogbn_products_simple_sagn_exponent_no_labels.sh
│   │   │   ├── train_ogbn_products_simple_sagn_exponent_use_labels.sh
│   │   │   ├── train_ogbn_products_simple_sagn_uniform_no_labels.sh
│   │   │   └── train_ogbn_products_simple_sagn_uniform_use_labels.sh
│   │   ├── ppi
│   │   │   └── train_ppi_sagn.sh
│   │   ├── ppi_large
│   │   │   └── train_ppi_large_sagn.sh
│   │   ├── reddit
│   │   │   └── train_reddit_sagn.sh
│   │   └── yelp
│   │       └── train_yelp_sagn.sh
│   └── src
│       ├── dataset.py
│       ├── gen_models.py
│       ├── layers.py
│       ├── models.py
│       ├── post_process.py
│       ├── pre_process.py
│       ├── __pycache__
│       │   ├── dataset.cpython-37.pyc
│       │   ├── gen_models.cpython-37.pyc
│       │   ├── layers.cpython-37.pyc
│       │   ├── models.cpython-37.pyc
│       │   ├── pre_process.cpython-37.pyc
│       │   ├── train_process.cpython-37.pyc
│       │   └── utils.cpython-37.pyc
│       ├── sagn.py
│       ├── train_process.py
│       └── utils.py
├── self-kd
│   ├── loss.py
│   ├── models.py
│   ├── model_utils.py
│   ├── train_kd.py
│   └── utils.py
├── sgc
│   ├── main.py
│   ├── __pycache__
│   │   └── utils.cpython-37.pyc
│   └── utils.py
├── tmp
│   ├── emb.pt
│   ├── y_soft_conv.pt
│   ├── y_soft.pt
│   └── y_soft_sage.pt
├── tmp2
│   └── y_soft.pt
├── tmp3
│   └── y_soft.pt
└── tmp-pretrain
    └── y_soft.pt

58 directories, 196 files
```