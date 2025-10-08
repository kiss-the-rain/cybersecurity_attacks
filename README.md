# 基于 LSTM 的网络攻击三分类项目（PyTorch）

识别三种攻击类型：**Malware、DDoS、Intrusion**。项目包含：数据预处理、特征工程、LSTM 模型、训练/验证/测试评估、可视化。

## 目录结构
```
cybersec_lstm_project/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ main.py                # 一键运行：预处理 -> 训练 -> 测试 -> 可视化
│  ├─ dataset.py             # 数据读取与预处理（中文注释）
│  ├─ model.py               # LSTM 模型定义
│  ├─ train.py               # 训练与验证循环
│  ├─ evaluate.py            # 测试集评估（分类报告）
│  └─ utils/
│     ├─ metrics.py          # 度量指标封装
│     └─ plotting.py         # 训练曲线可视化
└─ data/
   └─ cybersecurity_attacks.csv   # 请把你的CSV放到此处（或通过 --csv_path 参数指定）
```

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行（默认按 7:1:2 划分数据集）
```bash
python -m src.main --csv_path ./data/cybersecurity_attacks.csv --epochs 10
```

3. 主要可调参数（命令行）
- `--csv_path`: 数据集 CSV 路径
- `--batch_size`: 批大小（默认 64）
- `--epochs`: 训练轮数（默认 10）
- `--max_seq_len`: 文本最大序列长度（默认 40）
- `--embed_dim`: 词向量维度（默认 100）
- `--hidden_dim`: LSTM 隐藏单元（默认 128）
- `--lr`: 学习率（默认 1e-3）
- `--device`: 计算设备（`auto` 会优先选择可用的 CUDA 或 Apple MPS，例如 M3 Pro GPU）

> 注：所有源码中的**注释均为中文**，变量名**均为英文**，满足你的作业要求。
