"""
可视化工具
- 中文注释；变量名为英文
"""
import matplotlib.pyplot as plt

def plot_curves(history, out_path: str = None):
    """绘制训练/验证损失与准确率曲线"""
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    plt.figure(figsize=(10,4))
    # 左图：损失
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Train Loss")
    plt.plot(range(1, len(val_loss)+1), val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss"); plt.legend()

    # 右图：准确率
    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_acc)+1), train_acc, label="Train Acc")
    plt.plot(range(1, len(val_acc)+1), val_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training & Validation Accuracy"); plt.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
