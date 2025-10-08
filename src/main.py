"""
一键主程序
- 中文注释；变量名为英文
- 步骤：加载并预处理数据 -> 构建模型 -> 训练/验证 -> 测试 -> 可视化
"""
import argparse
import torch
from .dataset import load_and_preprocess
from .model import CyberAttackLSTM
from .train import build_loaders, train_epochs
from .evaluate import evaluate_on_test
from .utils.plotting import plot_curves

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM-based Cyber Attack Classification")
    parser.add_argument("--csv_path", type=str, default="./data/cybersecurity_attacks.csv", help="CSV数据路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--max_seq_len", type=int, default=40, help="文本最大序列长度")
    parser.add_argument("--embed_dim", type=int, default=100, help="词向量维度")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM隐藏单元数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="计算设备：auto/cpu/cuda/mps")
    parser.add_argument("--save_plot", type=str, default="", help="训练曲线保存路径（为空则直接显示）")
    return parser.parse_args()

def resolve_device(device_arg: str) -> torch.device:
    """根据用户输入自动解析设备，优先使用 GPU/MPS"""
    normalized = device_arg.lower()

    def mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        print("[WARN] CUDA 不可用，回退到 CPU。")
        return torch.device("cpu")

    if normalized == "mps":
        if mps_available():
            return torch.device("mps")
        print("[WARN] MPS 不可用，回退到 CPU。")
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    print(f"[WARN] 未识别的 device={device_arg}，回退到 CPU。")
    return torch.device("cpu")

def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 1) 加载与预处理
    (train_seq, train_static, y_train,
     val_seq, val_static, y_val,
     test_seq, test_static, y_test,
     vocab, info) = load_and_preprocess(args.csv_path, max_seq_len=args.max_seq_len)

    print(f"[INFO] Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"[INFO] Vocab Size: {info['vocab_size']}, Static Dim: {info['static_dim']}")

    # 2) 构建 DataLoader
    train_loader, val_loader = build_loaders(train_seq, train_static, y_train,
                                             val_seq, val_static, y_val,
                                             batch_size=args.batch_size)

    # 3) 构建模型
    model = CyberAttackLSTM(vocab_size=info["vocab_size"],
                            embed_dim=args.embed_dim,
                            hidden_dim=args.hidden_dim,
                            static_dim=info["static_dim"],
                            num_classes=3)

    # 4) 训练与验证
    history = train_epochs(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # 5) 测试评估
    evaluate_on_test(model, test_seq, test_static, y_test, batch_size=args.batch_size, device=device,
                     target_names=["Malware", "DDoS", "Intrusion"])

    # 6) 可视化
    if args.save_plot:
        plot_curves(history, out_path=args.save_plot)
        print(f"[INFO] Curves saved to: {args.save_plot}")
    else:
        plot_curves(history, out_path=None)

if __name__ == "__main__":
    main()
