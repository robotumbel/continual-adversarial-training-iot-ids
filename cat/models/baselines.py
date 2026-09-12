"""
baselines.py — DNN, RNN, LSTM baseline models.
All receive input [batch, seq_len, n_features] for fair comparison.
Output dict matches AAM-TRANS format: {'logits', 'anomaly_score', 'embeddings'}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNNModel(nn.Module):
    """
    Deep Neural Network baseline.
    Mean-pools the sequence dimension, then applies a 3-layer MLP.
    """
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_classes),
        )
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, features]
        pooled  = x.mean(dim=1)          # [batch, features]
        embed   = self.encoder(pooled)    # [batch, hidden//2]
        logits  = self.classification_head(embed)
        anomaly = self.anomaly_scorer(embed)
        return {"logits": logits, "anomaly_score": anomaly, "embeddings": embed}


class RNNModel(nn.Module):
    """
    Vanilla RNN baseline.
    Processes [batch, seq_len, features]; uses last hidden state.
    """
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, features]
        out, _ = self.rnn(x)            # [batch, seq_len, hidden]
        embed   = out[:, -1, :]         # last timestep [batch, hidden]
        logits  = self.classification_head(embed)
        anomaly = self.anomaly_scorer(embed)
        return {"logits": logits, "anomaly_score": anomaly, "embeddings": embed}


class LSTMModel(nn.Module):
    """
    LSTM baseline.
    Processes [batch, seq_len, features]; uses last hidden state.
    """
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, features]
        out, (hn, _) = self.lstm(x)     # out: [batch, seq_len, hidden]
        embed   = hn[-1]                # last layer's hidden [batch, hidden]
        logits  = self.classification_head(embed)
        anomaly = self.anomaly_scorer(embed)
        return {"logits": logits, "anomaly_score": anomaly, "embeddings": embed}
