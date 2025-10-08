"""
LSTM 模型定义模块
- 中文注释；变量名为英文
- 文本序列经 Embedding + LSTM 提取表示；与静态特征拼接后，全连接分类为三类
"""
from torch import nn
import torch

class CyberAttackLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, static_dim: int, num_classes: int = 3):
        super(CyberAttackLSTM, self).__init__()
        # 嵌入层：将token id映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM：提取序列特征（batch_first=True 以 [B, T, E] 形式输入）
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 全连接分类头：将序列表示与静态特征拼接后进行分类
        self.fc1 = nn.Linear(hidden_dim + static_dim, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, seq, static_feats):
        # seq: [B, T] 文本序列ID；static_feats: [B, S] 静态特征
        emb = self.embedding(seq)               # [B, T, E]
        _, (h, _) = self.lstm(emb)              # h: [num_layers, B, H]
        last_hidden = h[-1]                     # [B, H] 取最后一层隐藏状态
        combined = torch.cat([last_hidden, static_feats], dim=1)  # [B, H+S]
        x = self.fc1(combined)
        x = self.act(x)
        logits = self.fc2(x)
        return logits
