"""
LSTM 模型定义模块
- 中文注释；变量名为英文
- 文本序列经 Embedding + LSTM 提取表示；与静态特征拼接后，全连接分类为三类
"""
from torch import nn
import torch


class CyberAttackLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        static_dim: int,
        num_classes: int = 3,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fusion_hidden: int = 128,
        static_hidden: int = 0,
    ):
        super(CyberAttackLSTM, self).__init__()
        # 嵌入层：将token id映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(p=dropout)

        # LSTM：使用多层双向结构提取序列特征
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 静态特征前馈模块，可自动适配没有静态特征的情况
        self.has_static = static_dim > 0
        if self.has_static:
            static_hidden_dim = (
                static_hidden
                if static_hidden > 0
                else max(32, min(256, static_dim * 2))
            )
            self.static_net = nn.Sequential(
                nn.LayerNorm(static_dim),
                nn.Linear(static_dim, static_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            static_out_dim = static_hidden_dim
        else:
            self.static_net = None
            static_out_dim = 0

        fusion_in_dim = hidden_dim * 2 + static_out_dim
        fusion_hidden = max(fusion_hidden, num_classes)

        # 分类头：LayerNorm + Dropout 提高泛化
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, seq, static_feats):
        # seq: [B, T] 文本序列ID；static_feats: [B, S] 静态特征
        emb = self.embedding(seq)               # [B, T, E]
        emb = self.embedding_dropout(emb)
        _, (h, _) = self.lstm(emb)              # h: [num_layers * num_directions, B, H]

        # 双向 LSTM 取最后一层正向 + 反向隐藏状态
        if self.lstm.bidirectional:
            last_hidden = torch.cat((h[-2], h[-1]), dim=1)  # [B, 2H]
        else:
            last_hidden = h[-1]  # [B, H]

        if self.has_static and static_feats.numel() > 0:
            static_repr = self.static_net(static_feats)
            fused = torch.cat([last_hidden, static_repr], dim=1)
        else:
            fused = last_hidden

        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits
