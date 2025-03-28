import os
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

# TransformerModelクラスの定義
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

# PositionalEncodingクラスの定義
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

# 動画データセットクラス
class FingerSpellingDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_paths = []
        self.labels = []
        
        for label in os.listdir(video_dir):
            label_dir = os.path.join(video_dir, label)
            if os.path.isdir(label_dir):
                for video_name in os.listdir(label_dir):
                    if video_name.endswith(".mp4"):
                        self.video_paths.append(os.path.join(label_dir, video_name))
                        self.labels.append(label)
        
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(set(self.labels))}

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video, _, _ = read_video(video_path)
        video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        # クリッピングを適用
        clip_length = 10  # クリッピングするフレーム数
        if video.size(0) > clip_length:
            video = video[:clip_length]
        else:
            # パディングを適用
            pad_size = clip_length - video.size(0)
            video = torch.cat([video, torch.zeros(pad_size, video.size(1), video.size(2), video.size(3))])
        
        if self.transform:
            video = video.to(torch.float32) / 255.0
            video = self.transform(video)
        
        label_idx = self.label_map[label]
        return video, label_idx

# 前処理の定義
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データセットとデータローダーの作成
video_dir = './Video/transformer訓練'
batch_size = 4
dataset = FingerSpellingDataset(video_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルの定義
input_dim = 3 * 224 * 224  # 入力データのチャネル数、高さ、幅を掛け合わせた値
num_classes = len(dataset.label_map)
model = TransformerModel(input_dim, num_layers=4, num_heads=8, hidden_dim=256, output_dim=num_classes)

# 事前学習用の損失関数と最適化手法の定義
criterion_pretrain = torch.nn.MSELoss()
optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=0.001)

# 事前学習ループ
num_epochs_pretrain = 80
for epoch in range(num_epochs_pretrain):
    for batch, _ in dataloader:
        # バッチの前処理
        batch = batch.squeeze(0)  # 余分な次元を削除
        batch_size, seq_len, channels, height, width = batch.shape
        batch = batch.reshape(batch_size, seq_len, -1)  # (B, T, C*H*W)
        
        # 予測と損失の計算
        outputs = model(batch)
        loss = criterion_pretrain(outputs, batch)
        
        # 勾配の計算と更新
        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
    
    print(f"Pretraining Epoch [{epoch+1}/{num_epochs_pretrain}], Loss: {loss.item():.4f}")

# 事前学習済みモデルの保存
torch.save(model.state_dict(), "pretrained_finger_spelling_transformer.pth")

# 分類タスクの損失関数と最適化手法の定義
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 50
for epoch in range(num_epochs):
    for batch, labels in dataloader:
        # バッチの前処理
        batch = batch.squeeze(0)  # 余分な次元を削除
        batch_size, seq_len, channels, height, width = batch.shape
        batch = batch.reshape(batch_size, seq_len, -1)  # (B, T, C*H*W)
        
        # 予測と損失の計算
        outputs = model(batch)
        outputs = outputs.view(batch_size, -1)  # 出力を(バッチサイズ, クラス数)の形状に変換
        loss = criterion(outputs, labels)
        
        # 勾配の計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Fine-tuning Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# モデルの保存
torch.save(model.state_dict(), "finger_spelling_transformer.pth")