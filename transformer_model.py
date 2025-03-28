import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_pretrain = nn.Linear(hidden_dim, input_dim)  # 事前学習用の全結合層を追加
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        if self.training:
            x = self.fc_pretrain(x)  # 事前学習時は入力と同じサイズに変換
            return x
        else:
            x = x.mean(dim=1)  # 時系列方向に平均化
            x = self.fc(x)
            return x
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x