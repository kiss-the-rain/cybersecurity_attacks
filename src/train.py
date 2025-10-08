\
# -*- coding: utf-8 -*-
"""
训练与验证循环
- 中文注释；变量名为英文
"""
from typing import Tuple, List
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

def build_loaders(train_seq, train_static, y_train,
                  val_seq, val_static, y_val,
                  batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """构建训练/验证 DataLoader"""
    train_ds = TensorDataset(
        torch.tensor(train_seq, dtype=torch.long),
        torch.tensor(train_static, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(val_seq, dtype=torch.long),
        torch.tensor(val_static, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_epochs(model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3, device: str = "cpu"):
    """训练若干轮并返回历史曲线"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    for ep in range(1, epochs + 1):
        # 训练阶段
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for seq_b, static_b, y_b in train_loader:
            seq_b = seq_b.to(device)
            static_b = static_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(seq_b, static_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seq_b.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证阶段
        model.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seq_b, static_b, y_b in val_loader:
                seq_b = seq_b.to(device)
                static_b = static_b.to(device)
                y_b = y_b.to(device)

                logits = model(seq_b, static_b)
                loss = criterion(logits, y_b)

                val_total_loss += loss.item() * seq_b.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)

        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Epoch {ep}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return {
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "val_loss": val_loss_hist,
        "val_acc": val_acc_hist
    }
