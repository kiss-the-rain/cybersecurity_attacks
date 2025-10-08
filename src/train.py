"""
训练与验证循环
- 中文注释；变量名为英文
"""
from typing import Tuple, List, Union, Optional
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn.utils import clip_grad_norm_

def build_loaders(train_seq, train_static, y_train,
                  val_seq, val_static, y_val,
                  batch_size: int = 64) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
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

    class_weights: Optional[torch.Tensor] = None
    if len(train_ds) > 0:
        labels = train_ds.tensors[2]
        num_classes = int(labels.max().item() + 1)
        counts = torch.bincount(labels, minlength=num_classes).float()
        inv_freq = counts.sum() / (counts + 1e-6)
        class_weights = inv_freq / inv_freq.sum() * num_classes
    return train_loader, val_loader, class_weights

def train_epochs(model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3,
                 device: Union[str, torch.device] = "cpu",
                 class_weights: Optional[torch.Tensor] = None,
                 weight_decay: float = 1e-5,
                 max_grad_norm: float = 5.0):
    """训练若干轮并返回历史曲线"""
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.to(device_obj)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device_obj) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    train_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    best_state = None
    best_val_acc = -float("inf")
    lr_hist: List[float] = []

    for ep in range(1, epochs + 1):
        # 训练阶段
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for seq_b, static_b, y_b in train_loader:
            seq_b = seq_b.to(device_obj)
            static_b = static_b.to(device_obj)
            y_b = y_b.to(device_obj)

            optimizer.zero_grad()
            logits = model(seq_b, static_b)
            loss = criterion(logits, y_b)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
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
                seq_b = seq_b.to(device_obj)
                static_b = static_b.to(device_obj)
                y_b = y_b.to(device_obj)

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

        current_lr = optimizer.param_groups[0]["lr"]
        lr_hist.append(current_lr)
        print(f"Epoch {ep}: LR={current_lr:.5f}, Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "val_loss": val_loss_hist,
        "val_acc": val_acc_hist,
        "lr": lr_hist,
    }
