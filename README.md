# MAXP_DGL_Graph
[ 2021 MAXP 命题赛 任务一：[基于DGL的图机器学习任务 ](https://www.biendata.xyz/competition/maxp_dgl/) ] | [ 队伍: Graph@ICT ]



# MAXP竞赛——DGL图数据Baseline模型

本代码库是为2021 MAXP竞赛的DGL图数据所准备的Baseline模型，供参赛选手参考学习使用DGL来构建GNN模型。

代码库包括2个部分：
---------------

1. 用于数据预处理的4个Jupyter Notebook
2. 用DGL构建的3个GNN模型(GCN,GraphSage和GAT)，以及训练模型所用的代码和辅助函数。

依赖包：
------

- dgl==0.7.1
- pytorch==1.7.0
- pandas
- numpy
- datetime

如何运行：
-------

对于4个Jupyter Notebook文件，请使用Jupyter环境运行，并注意把其中的竞赛数据文件所在的文件夹替换为你自己保存数据文件的文件夹。
并记录下你处理完成后的数据文件所在的位置，供下面模型训练使用。

对于node2vec
```bash
cd node2vec/
nohup python main.py > ../outputs/node2vec.log 2>&1 &
tail -f ../outputs/node2vec.log
```
结果保存在`../dataset/emb.pt`中


对于GNN的模型，需要先cd到gnn目录，然后运行：

```bash
cd gnn/
python csv_idx_map.py
python model_train.py --GPU 1
# or run in backward
nohup python3 model_train.py --GPU 1 --use_emb --use_label --all_train > ../outputs/train1.log 2>&1 &
# check the result in terminal
tail -f ../outputs/train1.log
# ----------
# pretrain for cs
nohup python3 model_train.py --GPU 1 --use_emb --save_emb --all_train > ../outputs/train1.log 2>&1 &
cd ../correct_and_smooth
python3 main.py

# or
python3 model_train.py --data_path ../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 1 --out_path ./
```

*注意*：请把--data_path的路径替换成用Jupyter Notebook文件处理后数据所在的位置路径。其余的参数，请参考model_train.py里面的入参说明修改。

如果希望使用单GPU进行模型训练，则需要修改入参 `--GPU`的输入值为单个GPU的编号，如：

```bash
--GPU 0
```

如果希望使用单机多GPU进行模型训练，则需要修改入参 `--GPU`的输入值为多个可用的GPU的编号，并用空格分割，如：

```bash
--GPU 0 1 2 3
```

## 如何调试
进入gnn目录，配置`launch.json`，如下面所示（解析参数），点击开始调试。
```json
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
                "./"
            ]
        }
    ]
}
```

## Tree
```bash
.
├── dataset
│   ├── csv_idx_map.pkl
│   ├── diff_nodes.csv
│   ├── features.npy
│   ├── graph.bin
│   ├── IDandLabels.csv
│   ├── labels.pkl
│   ├── link_phase1.csv
│   ├── sample_submission_for_validation.csv
│   ├── test_id_dict.pkl
│   ├── train_nodes.csv
│   └── validation_nodes.csv
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
├── outputs
│   ├── node2vec.log
│   ├── submit_2021-10-13.csv
│   ├── submit_2021-10-14.csv
│   ├── submit_2021-10-15.csv
│   ├── train1.log
│   └── train.log
├── README 2.md
└── README.md

6 directories, 41 files
```