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

对于GNN的模型，需要先cd到gnn目录，然后运行：

```bash
cd gnn/
python csv_idx_map.py
python model_train.py --GPU 1

# or
python model_train.py --data_path ../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 1 --out_path ./
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