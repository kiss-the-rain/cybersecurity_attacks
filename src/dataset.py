"""
数据读取与预处理模块
- 中文注释；变量名为英文
- 读取CSV，处理缺失值/类别编码/数值标准化
- 文本清洗、词汇表构建、文本转ID序列
- 划分训练/验证/测试（7:1:2）
"""
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_text(s: str) -> str:
    """文本清洗：转小写并移除非字母数字字符"""
    s = str(s).lower()
    return re.sub(r"[^a-z0-9\s]", " ", s)

def build_vocab(train_series):
    """基于训练集构建词汇表（含 <PAD>=0, <UNK>=1）"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence in train_series:
        for w in sentence.split():
            if w and w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def text_to_ids(series, vocab, max_len=40):
    """将文本序列转为定长ID序列（截断/填充）"""
    ids_list = []
    pad_id = vocab["<PAD>"]
    unk_id = vocab["<UNK>"]
    for s in series:
        words = s.split()
        ids = [vocab.get(w, unk_id) for w in words[:max_len]]
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        ids_list.append(ids)
    return np.array(ids_list, dtype=np.int64)

def load_and_preprocess(
    csv_path: str,
    max_seq_len: int = 40,
    random_state: int = 42
):
    """读取CSV并完成预处理与数据划分，返回numpy数组与辅助对象"""
    # 1) 读取CSV
    data = pd.read_csv(csv_path)

    # 2) 缺失值相关的指示标志（示例：恶意软件迹象/IDS警报/数据包类型）
    if "Malware Indicators" in data.columns:
        data["malware_flag"] = data["Malware Indicators"].notna().astype(int)
    if "IDS/IPS Alerts" in data.columns:
        data["ids_flag"] = data["IDS/IPS Alerts"].notna().astype(int)
    if "Packet Type" in data.columns:
        data["packet_type_flag"] = (data["Packet Type"] == "Data").astype(int)

    # 3) 删除高基数或不参与训练的列（按示例字段名，若不存在则忽略）
    drop_cols = [
        "Timestamp", "Source IP Address", "Destination IP Address",
        "User Information", "Device Information", "Network Segment",
        "Geo-location Data", "Proxy Information", "Firewall Logs",
        "Malware Indicators", "IDS/IPS Alerts", "Packet Type", "Alerts/Warnings",
        "Source Port", "Destination Port"
    ]
    existing_drop_cols = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=existing_drop_cols, errors="ignore")

    # 4) 标签编码（Malware=0, DDoS=1, Intrusion=2）
    label_map = {"Malware": 0, "DDoS": 1, "Intrusion": 2}
    data["label"] = data["Attack Type"].map(label_map)

    # 5) 7:1:2 划分
    X = data.drop(columns=["Attack Type", "label"])
    y = data["label"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=random_state
    )

    # 6) 分离文本列（Payload Data），保留静态特征
    text_col = "Payload Data"
    train_text = X_train[text_col].astype(str).fillna("") if text_col in X_train.columns else pd.Series([""]*len(X_train))
    val_text = X_val[text_col].astype(str).fillna("") if text_col in X_val.columns else pd.Series([""]*len(X_val))
    test_text = X_test[text_col].astype(str).fillna("") if text_col in X_test.columns else pd.Series([""]*len(X_test))
    X_train = X_train.drop(columns=[text_col], errors="ignore")
    X_val = X_val.drop(columns=[text_col], errors="ignore")
    X_test = X_test.drop(columns=[text_col], errors="ignore")

    # 7) 类别特征独热编码（如 Protocol、Traffic Type、Action Taken 等）
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    merged_static = pd.get_dummies(pd.concat([X_train, X_val, X_test], ignore_index=True), columns=cat_cols)
    merged_static = merged_static.fillna(0)
    n_train = len(X_train)
    n_val = len(X_val)
    train_static = merged_static.iloc[:n_train].copy()
    val_static = merged_static.iloc[n_train:n_train+n_val].copy()
    test_static = merged_static.iloc[n_train+n_val:].copy()

    # 8) 数值标准化（排除独热后的二值列，仅对连续数值列进行标准化）
    potential_num_cols = [
        c for c in merged_static.columns
        if is_numeric_dtype(merged_static[c])
    ]
    num_cols = [
        c for c in potential_num_cols
        if merged_static[c].nunique(dropna=False) > 2
    ]
    scaler = StandardScaler()
    if num_cols:
        train_static[num_cols] = scaler.fit_transform(train_static[num_cols])
        val_static[num_cols] = scaler.transform(val_static[num_cols])
        test_static[num_cols] = scaler.transform(test_static[num_cols])

    # 9) 文本清洗 + 词汇表 + 转ID
    train_text = train_text.apply(clean_text)
    val_text = val_text.apply(clean_text)
    test_text = test_text.apply(clean_text)

    vocab = build_vocab(train_text)
    train_seq = text_to_ids(train_text, vocab, max_len=max_seq_len)
    val_seq = text_to_ids(val_text, vocab, max_len=max_seq_len)
    test_seq = text_to_ids(test_text, vocab, max_len=max_seq_len)

    # 10) 转 numpy
    train_static = train_static.to_numpy(dtype=np.float32)
    val_static = val_static.to_numpy(dtype=np.float32)
    test_static = test_static.to_numpy(dtype=np.float32)

    y_train = y_train.to_numpy(dtype=np.int64)
    y_val = y_val.to_numpy(dtype=np.int64)
    y_test = y_test.to_numpy(dtype=np.int64)

    info = {
        "vocab_size": len(vocab),
        "static_dim": train_static.shape[1],
        "label_map": label_map,
        "scaler_used": bool(num_cols),
        "num_cols": num_cols,
    }
    return (train_seq, train_static, y_train,
            val_seq, val_static, y_val,
            test_seq, test_static, y_test,
            vocab, info)
