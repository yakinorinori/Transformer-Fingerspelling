import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import math
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd 

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

# Transformer Model for Time Series Data
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size=82, num_classes=10, seq_length=30, num_layers=2, nhead=5, dim_feedforward=512):
        super(TimeSeriesTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size, 0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        return output

# モデルのロード
model_path = "model_epoch_80.pth"
num_classes = 46  # クラス数
model = TimeSeriesTransformer(
    feature_size=108, num_classes=num_classes, seq_length=100, num_layers=2, nhead=4
)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# MediaPipeのハンド認識を初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# 前フレームの指先座標を保持する変数を初期化
previous_finger_tips = {
    "thumb_tip": None,
    "index_tip": None,
    "middle_tip": None,
    "ring_tip": None,
    "pinky_tip": None,
}

video_folder_path = "Video/評価実験"
# 仮名文字リスト（0から始まるインデックス）
kana_list = [
    'あ', 'い', 'う', 'え', 'お',
    'か', 'き', 'く', 'け', 'こ',
    'さ', 'し', 'す', 'せ', 'そ',
    'た', 'ち', 'つ', 'て', 'と',
    'な', 'に', 'ぬ', 'ね', 'の',
    'は', 'ひ', 'ふ', 'へ', 'ほ',
    'ま', 'み', 'む', 'め', 'も',
    'や', 'ゆ', 'よ',
    'ら', 'る', 'れ', 'ろ', 'わ' 'ん',
]

# ラベルマッピング
label_mapping = {i + 1: kana for i, kana in enumerate(kana_list)}

min_frames = 10  # モデルが必要とする最小フレーム数を設定
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"  # macOSの日本語フォントのパス
font = ImageFont.truetype(font_path, 30)

def calculate_angle(base, tip):
    dx = tip.x - base.x
    dy = tip.y - base.y
    angle = np.arctan2(dy, dx) * (180.0 / np.pi)  # ラジアンを度に変換
    return angle

def calculate_angle_at_vertex(pointA, pointB, pointC):
    # ベクトルの計算
    AB = np.array([pointB.x - pointA.x, pointB.y - pointA.y])  # AからBへのベクトル
    AC = np.array([pointC.x - pointA.x, pointC.y - pointA.y])  # AからCへのベクトル
    
    # 内積と大きさを計算
    dot_product = np.dot(AB, AC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_AC = np.linalg.norm(AC)

    # コサインを用いて角度を計算
    cos_angle = dot_product / (magnitude_AB * magnitude_AC)

    # コサインの値からアークコサインを用いて角度を求め、度に変換
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 値を[-1, 1]に制限して計算
    angle_deg = angle_rad * (180.0 / np.pi)

    return angle_deg

# 手の向きを判定する関数を追加
def determine_hand_orientation(hand_landmarks):
    thumb_mcp = hand_landmarks.landmark[5]  # 親指の先
    palm_center = hand_landmarks.landmark[17]  # 手首
    finger_mcps = [
        hand_landmarks.landmark[2],  # 親指
        hand_landmarks.landmark[5],  # 人差し指
        hand_landmarks.landmark[9], # 中指
        hand_landmarks.landmark[13], # 薬指
        hand_landmarks.landmark[17]  # 小指
    ]

    # 親指のz座標と他の指のz座標を比較
    thumb_mcp_x = thumb_mcp.x
    avg_finger_mcp_x = palm_center.x

    # カメラに向いている場合は 1, そうでない場合は 0
    if thumb_mcp_x > avg_finger_mcp_x:
        return 1  # 表
    else:
        return 0  # 裏
    
# x-z平面の角度を計算する関数を追加
def calculate_xz_angle(base, tip):
    dx = tip.x - base.x
    dz = tip.z - base.z
    angle = np.arctan2(dz, dx) * (180.0 / np.pi)
    return angle

output_folder = "評価実験/transformer実験"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
# 効率的なビデオ処理
results_list = []  # List to store results for CSV
for filename in os.listdir(video_folder_path):
    if filename.endswith(('.mov', '.mp4')):  # MOVとMP4ファイルを処理 
        video_path = os.path.join(video_folder_path, filename)
        print(f"\n現在処理中の動画ファイル: {filename}\n")  # 動画ファイル名を出力
        cap = cv2.VideoCapture(video_path)
        buffer = deque(maxlen=min_frames)
        current_label = "None"
        frame_counter = 0

        while True:
            success, img = cap.read()
            if not success:
                print(f"ファイル {filename} の読み込み終了\n")
                break

            img = cv2.flip(img, 1)  # 左右を反転
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            hand_data = []

            # 手のランドマークが検出された場合
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 手のランドマークの座標を取得
                    lm_reference = hand_landmarks.landmark[0]  # 手首を基準点に設定
                    # 各ランドマークの座標を取得
                    wrist = hand_landmarks.landmark[0]  # 手首
                    palm_center = hand_landmarks.landmark[9]  # 手のひらの中心（手首の座標を使用）
                    thumb_tip = hand_landmarks.landmark[4]  # 親指の先
                    index_tip = hand_landmarks.landmark[8]  # 人差し指の先
                    middle_tip = hand_landmarks.landmark[12]  # 中指の先
                    ring_tip = hand_landmarks.landmark[16]  # 薬指の先
                    pinky_tip = hand_landmarks.landmark[20]  # 小指の先
                    
                    thumb_ip = hand_landmarks.landmark[3]  # 親指のMCP
                    index_pip = hand_landmarks.landmark[7]  # 人差し指の第二関節
                    middle_pip = hand_landmarks.landmark[11]  # 中指の第二関節
                    ring_pip = hand_landmarks.landmark[15]  # 薬指の第二関節
                    pinky_pip = hand_landmarks.landmark[19]  # 小指の第二関節
                    
                    # 角度ようの平面
                    thumb_mcp = hand_landmarks.landmark[2]  # 親指の先
                    index_mcp = hand_landmarks.landmark[5]  # 人差し指の先
                    middle_mcp = hand_landmarks.landmark[9]  # 人差し指の先
                    pinky_mcp = hand_landmarks.landmark[17]  # 小指の先
                    
                    # 人差し指と中指の第二関節のz座標を取得
                    index_z = hand_landmarks.landmark[7].z  # 人差し指の先
                    middle_z = hand_landmarks.landmark[12].z  # 中指の先
                    # z座標で降順ソート（z座標が大きい＝手前にある）
                    sorted_fingers = sorted(
                        {"index_z": index_z, "middle_z": middle_z}.items(),  # z座標を辞書に格納
                        key=lambda x: x[1],  # z座標を基準にソート
                        reverse=True  # 降順
                    )
                    
                    # 最も手前にある指が人差し指なら1、それ以外なら0
                    result = 1 if sorted_fingers[0][0] == "index_z" else 0
                    
                    # 最終結果を hand_data に追加
                    hand_data.append(result)
                    # 現在の指先座標を辞書に格納
                    current_finger_tips = {
                        "thumb_tip": thumb_tip,
                        "index_tip": index_tip,
                        "middle_tip": middle_tip,
                        "ring_tip": ring_tip,
                        "pinky_tip": pinky_tip,
                    }
                    # 前フレームとの相対座標を計算
                    relative_finger_tips = {}
                    for finger, current_tip in current_finger_tips.items():
                        if previous_finger_tips[finger] is not None:
                            prev_tip = previous_finger_tips[finger]
                            relative_x = current_tip.x - prev_tip.x
                            relative_y = current_tip.y - prev_tip.y
                            relative_z = current_tip.z - prev_tip.z
                            relative_x = relative_x
                            relative_y = relative_y
                            relative_z = relative_z
                            relative_finger_tips[finger] = (relative_x, relative_y, relative_z)
                        else:
                            # 前フレームのデータがない場合は (0, 0, 0) を使用
                            relative_finger_tips[finger] = (0, 0, 0)
                    # 手データの構築
                    for lm in hand_landmarks.landmark:
                        normalized_x = lm.x - lm_reference.x  # 相対座標
                        normalized_y = lm.y - lm_reference.y  # 相対座標
                        normalized_z = lm.z - lm_reference.z  # Z座標を相対的に
                        hand_data.extend([normalized_x, normalized_y, normalized_z])
                    # 隣接指同士の距離の計算
                    adjacent_distances = {
                        "thumb_index": np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])),
                        "thumb_middle": np.linalg.norm(np.array([thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y])),
                        "thumb_ring": np.linalg.norm(np.array([thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y])),
                        "thumb_pinky": np.linalg.norm(np.array([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y])),
                        "index_middle": np.linalg.norm(np.array([index_tip.x - middle_tip.x, index_tip.y - middle_tip.y])),
                        "indextip_mcp": np.linalg.norm(np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])),
                        "indextip_mcp-dip": np.linalg.norm(np.array([middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y]) - np.array([middle_tip.x - middle_pip.x, middle_tip.y - middle_pip.y])),
                        "indexpip_middlepip": np.linalg.norm(np.array([index_pip.x - middle_pip.x, index_pip.y - middle_pip.y])),
                        # "middle_ring": np.linalg.norm(np.array([middle_tip.x - ring_tip.x, middle_tip.y - ring_tip.y])),
                        "middle_center": np.linalg.norm(np.array([middle_tip.x - palm_center.x, middle_tip.y - palm_center.y])),
                        "ring_pinky": np.linalg.norm(np.array([ring_tip.x - pinky_tip.x, ring_tip.y - pinky_tip.y])),
                    }
                    
                    # 角度の計算
                    thumb_angle_from_palm = calculate_angle(palm_center, thumb_tip)
                    index_angle_from_palm = calculate_angle(palm_center, index_tip)
                    middle_angle_from_palm = calculate_angle(palm_center, middle_tip)
                    ring_angle_from_palm = calculate_angle(palm_center, ring_tip)
                    pinky_angle_from_palm = calculate_angle(palm_center, pinky_tip)
                    
                    thumb_angle_at_vertex = calculate_angle_at_vertex(thumb_tip, thumb_mcp, pinky_mcp)
                    index_angle_at_vertex = calculate_angle_at_vertex(index_tip, thumb_mcp, pinky_mcp)
                    middle_angle_at_vertex = calculate_angle_at_vertex(middle_tip, thumb_mcp,pinky_mcp)
                    ring_angle_at_vertex = calculate_angle_at_vertex(ring_tip, thumb_mcp, pinky_mcp)
                    pinky_angle_at_vertex = calculate_angle_at_vertex(pinky_tip, thumb_mcp, pinky_mcp)
                    
                    thumb_angle_from_base = calculate_angle(thumb_ip, thumb_tip)
                    index_angle_from_base = calculate_angle(index_pip, index_tip)
                    middle_angle_from_base = calculate_angle(middle_pip, middle_tip)
                    ring_angle_from_base = calculate_angle(ring_pip, ring_tip)
                    pinky_angle_from_base = calculate_angle(pinky_pip, pinky_tip)
                    middle_mcp = hand_landmarks.landmark[9]  # 中指の付け根 (MCP)
                    middleMCP_angle_from_base = calculate_angle(wrist, middle_mcp)
                    # x-z平面の角度を計算
                    thumb_xz_angle = calculate_xz_angle(middle_mcp, thumb_mcp)
                    index_xz_angle = calculate_xz_angle(palm_center, index_pip)
                    middle_xz_angle = calculate_xz_angle(palm_center, middle_pip)
                    pinky_xz_angle = calculate_xz_angle(middle_mcp, pinky_pip)
                    # 各データを手データに追加
                    hand_data.extend(adjacent_distances.values())
                    hand_data.extend([
                        thumb_angle_from_palm,
                        index_angle_from_palm,
                        middle_angle_from_palm,
                        ring_angle_from_palm,
                        pinky_angle_from_palm,
                        thumb_angle_at_vertex,
                        index_angle_at_vertex,
                        middle_angle_at_vertex,
                        ring_angle_at_vertex,
                        pinky_angle_at_vertex,
                        thumb_angle_from_base,
                        index_angle_from_base,
                        middle_angle_from_base,
                        ring_angle_from_base,
                        pinky_angle_from_base,
                        middleMCP_angle_from_base,
                        thumb_xz_angle,
                        pinky_xz_angle
                    ])
                    # 各指先の相対座標を hand_data に追加
                    for finger, relative_coords in relative_finger_tips.items():
                        hand_data.extend(relative_coords)  # (x, y, z) を追加

                    # 手の向きを判定して hand_data に追加
                    orientation = determine_hand_orientation(hand_landmarks)
                    hand_data.append(orientation)  # 手の向き（0 または 1）を追加
                    # 現在の指先座標を次のフレームのために保存
                    previous_finger_tips = current_finger_tips

                    # # # 最終的なhand_dataのサイズを確認
                    # assert len(hand_data) == 99, f"hand_dataのサイズが不正です: {len(hand_data)}"  # デバッグ用アサーション
                    
                    # モデルへの入力データを準備
                    input_data = np.array(hand_data)

                    # バッファに入力データを追加
                    buffer.append(input_data)

                if len(buffer) == buffer.maxlen:
                    input_tensor = torch.tensor(np.array(buffer), dtype=torch.float).unsqueeze(1)
                    input_tensor = input_tensor.permute(1, 0, 2)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        current_label = label_mapping.get(predicted.item(), "Unknown")
                    # 現在のフレームでの認識結果を表示
                    print(f"フレームの認識結果: {current_label}")
                    
                    if current_label != "None":
                        frame_filename = os.path.join(output_folder, f"{frame_counter}_{current_label}.png")
                        cv2.imwrite(frame_filename, img)  # Save frame as PNG
                        frame_counter += 1  # Increment frame counte
                    buffer.clear()  # バッファをクリア
                    # 新しいデータを受け入れる準備
                    previous_finger_tips = {
                        "thumb_tip": None,
                        "index_tip": None,
                        "middle_tip": None,
                        "ring_tip": None,
                        "pinky_tip": None,
                    }
            # 現在の認識結果をフレームに表示 (Pillowを使用)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            text = f"認識結果: {current_label}"
            text_bbox = draw.textbbox((10, 10), text, font=font)

            # 認識結果表示部分の背景を黒にする（余分なスペースを追加）
            padding = 10
            background_bbox = (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding,
            )
            draw.rectangle(background_bbox, fill=(0, 0, 0))
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
            cv2.imshow("Video", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()

cv2.destroyAllWindows()
hands.close()