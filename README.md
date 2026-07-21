# SCA-GDA

## 工程结构

```text
.
├── train.py                  # 单一训练入口
├── src/
│   ├── data.py               # MATLAB 图数据加载与轻量 GraphData
│   ├── model.py              # GCN、对比学习、互信息损失、边界映射器
│   ├── scagda.py               # 类别演化与边界跟踪核心
│   └── metrics.py            # Accuracy / Macro-F1 / Micro-F1
├── scripts/
│   └── train_slurm.sh        # Slurm 集群提交脚本
├── data/                     # 数据目录；数据文件不提交 Git
├── requirements.txt
└── .gitignore
```

## 环境安装

建议使用 Python 3.9 或更高版本。GPU 环境应先按照本机 CUDA 版本安装匹配的 PyTorch，再安装其余依赖。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

该实现使用纯 PyTorch GCN，不强制依赖 `torch_geometric`。

## 数据格式

当前精简加载器支持原工程 Citation/Blog 数据使用的 MATLAB 格式。每个 `.mat` 文件需要包含：

- `attrb`：节点特征矩阵，形状 `[N, F]`；
- `network`：稀疏或稠密邻接矩阵，形状 `[N, N]`；
- `group`：类别索引或 one-hot 标签，形状 `[N]` 或 `[N, C]`。

将数据放入 `data/`，例如：

```text
data/dblpv7.mat
data/citationv1.mat
```


## 本地训练

```bash
python train.py \
  --source data/dblpv7.mat \
  --target data/citationv1.mat \
  --device cuda \
  --epochs 400 \
  --warmup-epochs 100 \
  --ramp-epochs 80 \
  --runs 10
```

CPU 调试：

```bash
python train.py \
  --source data/dblpv7.mat \
  --target data/citationv1.mat \
  --device cpu \
  --epochs 2 \
  --hidden-dim 32 \
  --contrast-max-nodes 256
```

## Slurm 集群运行

`scripts/train_slurm.sh` 中的 `#SBATCH`、`module` 和 `srun` 均是 **Slurm 集群指令**，不适用于普通本地终端。提交前必须根据所在集群修改分区名、GPU 资源、CUDA 模块、虚拟环境和数据路径。

```bash
sbatch scripts/train_slurm.sh
```

也可通过环境变量覆盖数据路径：

```bash
SOURCE_FILE=/path/source.mat \
TARGET_FILE=/path/target.mat \
OUTPUT_DIR=/path/outputs \
sbatch scripts/train_slurm.sh
```

## 输出

每个源域到目标域任务输出到：

```text
outputs/<source>_to_<target>/
├── aggregate.json
└── run_01/
    ├── checkpoint.pt
    ├── summary.json
    └── train_log.csv
```
