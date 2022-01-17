# MAXP-DGL-Graph-Machine-Learning-Challenge

![title](./title.png)

[ 2021 MAXP 命题赛 任务一：[基于DGL的图机器学习任务 ](https://www.biendata.xyz/competition/maxp_dgl/) ] | [ Team: *Graph@ICT* ]

[2021-MAXP-DGL图机器学习大赛解决方案-8-Graph@ICT.pdf](./2021-MAXP-DGL图机器学习大赛解决方案-8-Graph@ICT.pdf)

最终Test成绩（b榜）：rank6🥉，[[获奖名单](https://www.biendata.xyz/competition/maxp_dgl/winners/)] | [[答辩视频](https://www.bilibili.com/video/BV1fr4y1v737?p=2)]。

有任何问题请联系 chihuixuan21@mails.ucas.ac.cn.

P.S. [一位大佬整理的前排解决方案](https://github.com/CYBruce/MAXP-DGL-solutions)。

<!-- ![res](./result.png) -->

## 依赖包：

```bash
dgl-cu102==0.7.2
pytorch==1.7.0
sklearn
pandas
numpy
datetime
tqdm
...
```

## 环境安装
<!-- 依赖包见`requirement.txt`. -->

```bash
pip install -r requirement.txt
```

## GPU
+ Tesla V100 (32GB)

## 关键路径
```bash
.
|--dataset
|--outputs
|--GAMLP
|  |--scripts
|  |--output
|--gnn
|--node2vec
```


## 运行 Final Test 

**为了完整复现，我们建议从头进行端到端的训练（Scalable GNN训练的很快）**

### 0、准备工作和数据路径
```bash
cd MAXP_DGL_Graph/
mkdir dataset  # 存放所有数据集
mkdir outputs  # 存放输出结果
```
### 1、 运行jupyter进行数据预处理
```python
# 生成feature和DGL graph
process-1.ipynb
process-2.ipynb
process-3.ipynb
process-4.ipynb
# 根据test nodes构造sample_submission_for_test.csv
gen_test_submitcsv.ipynb
```

### 2、构造必要的id--index映射关系
```bash
cd gnn/
python3 csv_idx_map.py
```

### 3、运行node2vec得到所有节点的embedding
```bash
cd node2vec/
nohup python main.py > ../outputs/node2vec.log 2>&1 &  
```
大约需要12h。

也可以直接下载`node2vec_emb.zip`，放到`./dataset`中并解压。链接: [https://pan.baidu.com/s/1Kg3bLPJ6q8yUNjiPtQUBtA](https://pan.baidu.com/s/1Kg3bLPJ6q8yUNjiPtQUBtA)  密码: dvuw


### 4、训练node2vec特征增强的GAMLP (8-fold)

```bash
cd GAMLP/scripts/
nohup bash train_maxp_kfold.sh > ../output/gamlp.log 2>&1 &
```
大约需要16h。

也可以直接下载`gamlp_ckpt.zip`，放到`./dataset`中并解压，**解压后可以不跑第4步以及之前的代码，直接运行第5步**。链接: [https://pan.baidu.com/s/1Q3v0Hlwqu8qI9-SCNBjt4A](https://pan.baidu.com/s/1Q3v0Hlwqu8qI9-SCNBjt4A)  密码: tn8h

每一折的best model模型文件`model.zip`可以从此处下载，解压后放到`./GAMLP/output/maxp`中。链接: [https://pan.baidu.com/s/1JfYz0oUcVexxMoCZzsTqXQ](https://pan.baidu.com/s/1JfYz0oUcVexxMoCZzsTqXQ)  密码: 6e5b

### 5、C&S和模型融合
```bash
cd GAMLP/
nohup python3 ensemble.py --all_train --gpu 1 > ./output/ensem.log 2>&1 &
```
大约需要2h。



### 6、下载submit文件
最终test：`./outputs/submit_test_gamlp_ensem_xxxx-xx-xx_n2v.csv`

最终valid：`./outputs/submit_val_gamlp_ensem_xxxx-xx-xx_n2v.csv`

| Method | Score |
|:-:|:-:|
| GAMLP (leaky-relu, 9 hops, 8-fold) + node2vec + C&S (DAD, AD) + Model Merge (+GAMLP_8fold_seed_{0-2}) | 49.7822086481499 |
| GAMLP (leaky-relu, 9 hops, 8-fold) + C&S (DAD, AD) + Model Merge (+GAMLP_8fold_seed_{0-2}) | 49.7923833548815 |
| GAMLP (leaky-relu, 9 hops, 8-fold) + C&S (DAD, AD) + Model Merge (+GAMLP_8fold_seed_{0}) | 49.7767704428278 |  

----

## Valid提交结果

a榜（Valid）结果记录。a榜最终成绩为14。

|Date  | Method | Score |
|:-:|:-:|:-:|
| 2021-11-17 | GAMLP (leaky-relu, 9 hops, 8-fold) + node2vec + C&S (DAD, AD) + Model Merge (+GAMLP_8fold_seed_{0-1})  |  55.53829808307 |
| 2021-11-13 | GAMLP (leaky-relu, 9 hops) + node2vec + C&S (DAD, AD) + Model Merge (+GAMLP_seed_{0-9}) + lr adjustment | 55.5081350224647 |
| 2021-11-08 | GAMLP (leaky-relu) + node2vec + C&S + Model Merge (+GAMLP_seed_{0-9}) | 55.3825306357653 |
| 2021-11-06 | GAMLP + node2vec + C&S + Model Merge (+GAMLP_seed_{0-9}) | 55.3694749826675 |
| 2021-11-05| GAMLP + node2vec + C&S + Model Merge (+GAMLP_seed_1, +SAGN_SE_stage_0, +SAGN_SE_stage_2, +SAGE, +GAT)  | 55.0993580220235 |
| 2021-11-04| GAMLP + node2vec + C&S + Model Merge (+SAGN-SE, +SAGE, +GAT)  | 55.0070680604702 |
| 2021-10-31 | SAGN + node2vec + SE + Model Merge (+GAT, +SAGE) + C&S | 54.5420166932282 |
| 2021-10-24 | GAT + node2vec + FLAG + C&S + Model Merge (+SAGE, +GCN) | 54.2394856973069 |
| 2021-10-22 | GAT + node2vec + FLAG + C&S | 53.9846753644328 |
| ... | ... | ... |
| 2021-10-18 | GAT+res+bn+dropout+train_label的特征增强+FLAG  +增加训练epoch（5--10） | 52.95 |
| 2021-10-17 | GAT+res+bn+dropout+train_label的特征增强+FLAG  +inception（已注释） | 小于52.53 |
| 2021-10-15 | GAT+res+bn+dropout+train_label的特征增强  +FLAG | 52.53 |
| 2021-10-15 | GAT+res+bn+dropout  +train_label的特征增强 | 51.27 |
| 2021-10-14 | GAT+  res+bn+dropout，调整了采样策略 | 50.79 |
| 2021-10-13 | GraphSAGE   +一些tricks | 48.48 |
| 2021-10-13 | GraphSAGE（baseline） | 48.14 |

<!-- ## 代码库包括2个部分：


1. 用于数据预处理的4个Jupyter Notebook
2. 用DGL构建的3个GNN模型(GCN,GraphSage和GAT)，以及训练模型所用的代码和辅助函数。 -->



## 全部探索如下：

[查看项目的整个目录树.](#Tree)

平均 degree: 16.9152 (加上反向边和self-loop后)

### 运行jupyter进行数据预处理

```bash
cd MAXP_DGL_Graph/
mkdir dataset
mkdir outputs
```

对于4个Jupyter Notebook文件，请使用Jupyter环境运行，并注意把其中的竞赛数据文件所在的文件夹替换为你自己保存数据文件的文件夹。
并记录下你处理完成后的数据文件所在的位置，供下面模型训练使用。

### 运行node2vec并保存Embedding
```bash
cd node2vec/
# run node2vec in backward
nohup python main.py > ../outputs/node2vec.log 2>&1 &
tail -f ../outputs/node2vec.log
```
结果保存在`../dataset/emb.pt`中， 下载链接[emb.pt](https://drive.google.com/u/0/uc?id=1gvJQBTbl8sIaJND_tBVA3Ua2TcU-4Kfy&export=download) (1.7GB)。


### 运行SGC并保存Embedding
```bash
cd sgc/
python3 main.py
```
结果保存在`../dataset/sgc_emb.pt`中。



### 运行GNN模型并保存test结果
对于GNN的模型，需要先cd到gnn目录，然后运行：

```bash
cd gnn/
# generate index map
python csv_idx_map.py
# then run gnn in backward
nohup python3 model_train.py --GPU 1 --use_emb node2vec --use_label --flag --all_train > ../outputs/train1.log 2>&1 &
# check the result in terminal
tail -f ../outputs/train1.log

# or
python3 model_train.py --data_path ../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 1 --out_path ./
```
test结果保存在`../outputs/submit_xxxx-xx-xx.csv`中。

### 使用C&S方法并进行模型融合
目前融合GAT， GraphSAGE， GCN， 对logits进行加权求和。

```bash
# pretrain model in backward
cd gnn/
# graphattn
nohup python3 model_train.py --GPU 1 --use_emb node2vec --save_emb --all_train > ../outputs/train1.log 2>&1 &
# graphsage
nohup python3 model_train.py --GPU 0 --use_emb node2vec --save_emb --gnn_model graphsage --flag --all_train > ../outputs/train.log 2>&1 &
# graphconv
nohup python3 model_train.py --GPU 1 --use_emb node2vec --save_emb --gnn_model graphconv --flag --all_train > ../outputs/train1.log 2>&1 &
# run c&s
cd ../correct_and_smooth
python3 main.py --all_train
```
inference logits保存在`../dataset.y_soft.pt` (GAT), `../dataset.y_soft_sage.pt` (GraphSAGE), `../dataset.y_soft_conv.pt` (GCN)中。
test结果保存在`../outputs/submit_cs_xxxx-xx-xx.csv`中。

### SAGN+SE使用方法
进入`SAGN_with_SLE`文件夹，按照指示进行运行。
```bash
cd SAGN_with_SLE/scripts/maxp
nohup bash train_maxp_sagn_use_label.sh > ../../outputs/sagn.log 2>&1 &

# c&s + merge models
cd SAGN_with_SLE/src/
python3 post_process.py 
```
inference logits保存在`./SAGN_with_SLE/intermediate_outputs/maxp/sagn/use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_*_stage_*.pt`中。
test结果保存在`../outputs/submit_sagn_xxxx-xx-xx.csv`中。

### GAMLP+C&S+模型融合

```bash
cd GAMLP/scripts/
# run all trains with seed 0-9
nohup bash train_maxp_kfold.sh > ../output/gamlp.log 2>&1 &
# run k-fold cv with diff seeds
nohup bash train_maxp_all.sh > ../output/gamlp1.log 2>&1 &

# post process + ensemble (model merge and c&s)
cd GAMLP/
nohup python3 ensemble.py --all_train --gpu 0 > ./output/ensem.log 2>&1 &
```
inference logits保存在`../dataset/gamlp_{seed}.pt`(all train)和`../dataset/gamlp_{k}fold_seed{seed}.pt`(kfold)中。
<!-- All train的test结果保存在`../outputs/submit_gamlp_cs_xxxx-xx-xx.csv`中，k-fold的 -->
最终ensemble的test结果保存在`../outputs/submit_gamlp_ensem_xxxx-xx-xx.csv`中。

**模型融合 (所有模型先C&S)：**

K-fold cross-validation -- Soft Voting: (16 ensemble)
| model | weight |
|:-:|:-:|
| GAMLP (seed 0, fold 0) | 0.2 |
| GAMLP (seed 0, fold 1) | 0.2 |
| ... | ... |
| GAMLP (seed 1, fold 7) | 0.2 |

**CV Results**
8 fold cv (0-7), 3 seeds (0-2).
| model | Val acc (%) |
|:-:|:-:|
| GAMLP (seed 0, 8 fold) | 0.6062 ± 0.0014 |
| GAMLP (seed 1, 8 fold) | 0.6063 ± 0.0012 |
| GAMLP (seed 2, 8 fold) | 0.6061 ± 0.0013 |

| seed | Val Acc (%) |
|:-:|:-:|
| 0 | **0.6076**, 0.6036, 0.6060, **0.6088**, 0.6060, 0.6061, 0.6059, 0.6055 |
| 1 | 0.6066, 0.6036, **0.6066**, **0.6068**, **0.6076**, **0.6068**, 0.6055, **0.6069** |
| 2 | **0.6076**, 0.6033, **0.6066**, **0.6075**, 0.6057, 0.6066, 0.6058, 0.6058 |




### ~~ogbn-papers100M预训练~~
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

*注意*：请把--data_path的路径替换成用Jupyter Notebook文件处理后数据所在的位置路径。其余的参数，请参考model_train.py里面的入参说明修改。

如果希望使用单GPU进行模型训练，则需要修改入参 `--GPU`的输入值为单个GPU的编号，如：

```bash
--GPU 0
```

如果希望使用单机多GPU进行模型训练，则需要修改入参 `--GPU`的输入值为多个可用的GPU的编号，并用空格分割，如：

```bash
--GPU 0 1 2 3
```

### VSCode如何进行调试
进入gnn目录，配置`launch.json`，如下面所示（解析参数），uncomment相应的内容，点击开始调试。
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
            // "args": [
            //     // "--data_path",
            //     // "../dataset",
            //     // "--gnn_model",
            //     // "graphattn",
            //     // "--hidden_dim",
            //     // "64",
            //     // "--n_layers",
            //     // "2",
            //     // "--fanout",
            //     // "20,20",
            //     // "--batch_size",
            //     // "4096",
            //     "--GPU",
            //     "1",
            //     // "--out_path",
            //     // "../outputs",
            //     // "--use_emb",
            //     // "--save_emb",
            //     "--all_train"
            // ]

            // SAGN products
            // "args": [
            //     "--dataset",
            //     "ogbn-products",
            //     "--gpu",
            //     "0",
            //     "--aggr-gpu",
            //     "0",
            //     "--model",
            //     "sagn",
            //     "--seed",
            //     "0",
            //     "--num-runs",
            //     "10",
            //     "--threshold",
            //     "0.9",
            //     // "--epoch-setting",
            //     // "1000+200+200",
            //     "--lr",
            //     "0.001",
            //     "--weight-style",
            //     "uniform",
            //     "--batch-size",
            //     "50000",
            //     "--num-hidden",
            //     "512",
            //     "--dropout",
            //     "0.5",
            //     "--attn-drop",
            //     "0.4",
            //     "--input-drop",
            //     "0.2",
            //     "--K",
            //     "3",
            //     "--label-K",
            //     "9",
            //     "--use-labels",
            //     "--weight-decay",
            //     "0",
            //     "--warmup-stage",
            //     "-1",
            //     "--memory-efficient"
            // ]

            //SAGN maxp papers100M
            // "args": [
            //     "--dataset",
            //     "maxp", // ogbn-papers100M
            //     "--gpu",
            //     "0",
            //     "--aggr-gpu",
            //     "1",
            //     "--eval-every",
            //     "1",
            //     "--model",
            //     "sagn",
            //     "--zero-inits",
            //     "--chunks",
            //     "1",
            //     "--memory-efficient",
            //     "--load-embs",
            //     "--load-label-emb",
            //     "--seed",
            //     "0",
            //     "--num-runs",
            //     "1",
            //     "--threshold",
            //     "0.5",
            //     "--epoch-setting",
            //     "1",
            //     "1",
            //     // "1",
            //     // "1",
            //     "--lr",
            //     "0.001",
            //     "--batch-size",
            //     "5000",
            //     "--num-hidden",
            //     "1024",
            //     "--dropout",
            //     "0.5",
            //     "--attn-drop",
            //     "0.",
            //     "--input-drop",
            //     "0.0",
            //     "--label-drop",
            //     "0.5",
            //     "--K",
            //     "5",
            //     "--label-K",
            //     "9",
            //     // "--use-labels",
            //     "--weight-decay",
            //     "0",
            //     "--warmup-stage",
            //     "-1",
            //     "--all-train"
            // ]
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
│   ├── gamlp_1.pt
│   ├── gamlp_2.pt
│   ├── gamlp.pt
│   ├── graph.bin
│   ├── IDandLabels.csv
│   ├── labels.pkl
│   ├── link_phase1.csv
│   ├── sample_submission_for_validation.csv
│   ├── sgc_emb.pt
│   ├── test_id_dict.pkl
│   ├── train_nodes.csv
│   ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_0.pt
│   ├── use_labels_False_use_feats_True_K_5_label_K_9_probs_seed_0_stage_2.pt
│   ├── validation_nodes.csv
│   ├── y_soft_conv.pt
│   ├── y_soft.pt
│   └── y_soft_sage.pt
├── GAMLP
│   ├── cs.py
│   ├── data
│   │   ├── mag
│   │   ├── preprocess_papers100m.py
│   │   └── train_graph_emb.sh
│   ├── dataset
│   │   └── ogbn_products
│   │       ├── embedding
│   │       ├── mapping
│   │       │   ├── labelidx2productcategory.csv.gz
│   │       │   ├── nodeidx2asin.csv.gz
│   │       │   └── README.md
│   │       ├── processed
│   │       │   ├── dgl_data_processed
│   │       │   ├── geometric_data_processed.pt
│   │       │   ├── pre_filter.pt
│   │       │   └── pre_transform.pt
│   │       ├── raw
│   │       │   ├── edge.csv.gz
│   │       │   ├── node-feat.csv.gz
│   │       │   ├── node-label.csv.gz
│   │       │   ├── num-edge-list.csv.gz
│   │       │   └── num-node-list.csv.gz
│   │       ├── RELEASE_v1.txt
│   │       ├── split
│   │       │   └── sales_ranking
│   │       │       ├── test.csv.gz
│   │       │       ├── train.csv.gz
│   │       │       └── valid.csv.gz
│   │       └── tmp
│   ├── GAMLP.pdf
│   ├── heteo_data.py
│   ├── layer.py
│   ├── load_dataset.py
│   ├── logger.py
│   ├── mag_perf.png
│   ├── main.py
│   ├── model.py
│   ├── output
│   │   ├── gamlp.log
│   │   ├── maxp
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_0.pkl
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_0.pt
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_1.pkl
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_1.pt
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_2.pkl
│   │   │   ├── 038fec23d66646bdbea4f48321c06d45_2.pt
│   │   │   ├── 13f02d430d834005afea80f9e6030903_0.pkl
│   │   │   ├── 13f02d430d834005afea80f9e6030903_0.pt
│   │   │   ├── 13f02d430d834005afea80f9e6030903_1.pkl
│   │   │   ├── 23531c5263b3401aba302b1f73ba5247_0.pkl
│   │   │   ├── 2759c14d61b0450f9126dc477056ca57_0.pkl
│   │   │   ├── 2759c14d61b0450f9126dc477056ca57_0.pt
│   │   │   ├── 4a0fc3843a654473a3523997d3e83165_0.pkl
│   │   │   ├── 4a0fc3843a654473a3523997d3e83165_0.pt
│   │   │   ├── 5092dc80131349739ebb3f57aa58595d_0.pkl
│   │   │   ├── 5663c4882fe94b7e868ab256afdb7d6a_0.pkl
│   │   │   ├── 59cfbd78a816409db493374a3f6015aa_0.pkl
│   │   │   ├── 59cfbd78a816409db493374a3f6015aa_0.pt
│   │   │   ├── 5ab3c46610744dd5b56dbb567d0c8e8a_0.pkl
│   │   │   ├── 68229d24c6984b27964cf146a2e8a25c_0.pkl
│   │   │   ├── 698c305da278499ca72571a1df44f3b7_0.pkl
│   │   │   ├── 698c305da278499ca72571a1df44f3b7_0.pt
│   │   │   ├── 6a87a8e7364244b4bc7d080cb3881b93_0.pkl
│   │   │   ├── 6db31805325040f496c3620c82cda285_0.pkl
│   │   │   ├── 6fed9e2239be4e36bdd59f24a8c3fe75_0.pkl
│   │   │   ├── 8372dcdf20584876b9d330ec5b8d8583_0.pkl
│   │   │   ├── 8372dcdf20584876b9d330ec5b8d8583_0.pt
│   │   │   ├── 8bc7149252e4410ea4227ff6ef00fdd1_0.pkl
│   │   │   ├── 8e1ca6b74ced47e39c28134137ed4d76_0.pkl
│   │   │   ├── 8e1ca6b74ced47e39c28134137ed4d76_0.pt
│   │   │   ├── 9b85f6536a4f4a38a36f5dded21fbbe4_0.pkl
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_0.pkl
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_0.pt
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_1.pkl
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_1.pt
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_2.pkl
│   │   │   ├── a32689c83e474a3b81fa05b530e549cd_2.pt
│   │   │   ├── ab15635c9b534e25bd591dd7677a67de_0.pkl
│   │   │   ├── b0b6c19d319049e695201d4d82aef63e_0.pkl
│   │   │   ├── b0b6c19d319049e695201d4d82aef63e_0.pt
│   │   │   ├── d30ab2026fd6474db2a369c9e41c806b_0.pkl
│   │   │   ├── d30ab2026fd6474db2a369c9e41c806b_0.pt
│   │   │   ├── e1251971498a43d681daaa5f0d7cb4cf_0.pkl
│   │   │   ├── e19d270746b84de3a164e687736e71ca_0.pkl
│   │   │   ├── e19d270746b84de3a164e687736e71ca_0.pt
│   │   │   ├── e989e5b5b044441bbd0bbd3e49da9f9b_0.pkl
│   │   │   ├── e989e5b5b044441bbd0bbd3e49da9f9b_0.pt
│   │   │   ├── f52a0739125a4778940a40cfdf0e3900_0.pkl
│   │   │   ├── f52a0739125a4778940a40cfdf0e3900_0.pt
│   │   │   ├── gamlp_0.pt
│   │   │   └── output_0.npy
│   │   └── ogbn-products
│   ├── papers100M_perf.png
│   ├── post_process.py
│   ├── products_perf.png
│   ├── __pycache__
│   │   ├── heteo_data.cpython-37.pyc
│   │   ├── layer.cpython-37.pyc
│   │   ├── load_dataset.cpython-37.pyc
│   │   ├── model.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts
│   │   ├── train_maxp.sh
│   │   └── train_products.sh
│   └── utils.py
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
│   ├── submit_gamlp_2021-11-04.csv
│   ├── submit_gamlp_2021-11-05.csv
│   ├── submit_gamlp_cs_2021-11-04.csv
│   ├── submit_gamlp_cs_2021-11-05.csv
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

74 directories, 297 files
```