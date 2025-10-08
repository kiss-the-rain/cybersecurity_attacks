"""
测试集评估
- 中文注释；变量名为英文
"""
from typing import List, Union
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

def evaluate_on_test(model, test_seq, test_static, y_test, batch_size: int = 64,
                     device: Union[str, torch.device] = "cpu", target_names: List[str] = None):
    """在测试集上进行评估并打印分类报告"""
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.to(device_obj)
    test_ds = TensorDataset(
        torch.tensor(test_seq, dtype=torch.long),
        torch.tensor(test_static, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for seq_b, static_b, y_b in test_loader:
            seq_b = seq_b.to(device_obj)
            static_b = static_b.to(device_obj)
            y_b = y_b.to(device_obj)
            logits = model(seq_b, static_b)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_b.cpu().tolist())

    if target_names is None:
        target_names = ["Malware", "DDoS", "Intrusion"]
    print("Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
