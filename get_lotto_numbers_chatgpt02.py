import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from collections import defaultdict

# -------------------------------------------------
# 1) CSV 불러오기
# -------------------------------------------------
df = pd.read_csv('history_from_cafe.csv', header=None)
df.columns = [
    'ball',   # 공번호 (1~5)
    'round',  # 회차
    'n1','n2','n3','n4','n5','n6',
    'bonus'
]

# 메인 + 보너스 번호를 한 세트로 묶어두기 (set 또는 list)
df['numbers_set'] = df.apply(
    lambda row: set([row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6'], row['bonus']]),
    axis=1
)

# -------------------------------------------------
# 2) Dataset 구성 함수
#    (공번호별로 분리 → build_dataset_for_ball)
# -------------------------------------------------
def build_dataset_for_ball(sub_df, lookback=5):
    """
    sub_df : 특정 공번호(ball)에 해당하는 데이터프레임 (회차 오름차순 정렬 전제)
    lookback : 최근 몇 회차를 볼 것인지
    
    return: X, Y (numpy array)
      - X.shape = (samples, 45)   # 각 sample마다 번호1~45에 대한 가중 합
      - Y.shape = (samples, 45)   # 각 sample마다 실제로 등장한 번호(7개)에 1, 나머지 0
    """
    sub_df = sub_df.sort_values('round').reset_index(drop=True)
    sets_of_numbers = list(sub_df['numbers_set'])
    
    X_list = []
    Y_list = []
    
    for i in range(len(sub_df)):
        if i < lookback:
            continue
        
        # --- X: 이전 lookback 회에 대한 가중치 합 ---
        freq_vec = np.zeros(45, dtype=float)
        for offset in range(1, lookback+1):
            past_idx = i - offset
            weight = (lookback - offset + 1)  # 최근 회차일수록 큰 weight
            for num in sets_of_numbers[past_idx]:
                freq_vec[num-1] += weight
        
        # --- Y: 이번(i번째) 실제 등장 번호(7개) → 45차원 0/1
        label_vec = np.zeros(45, dtype=int)
        for num in sets_of_numbers[i]:
            label_vec[num-1] = 1
        
        X_list.append(freq_vec)
        Y_list.append(label_vec)
    
    X = np.array(X_list, dtype=np.float32)  # (samples, 45)
    Y = np.array(Y_list, dtype=np.float32)  # (samples, 45)
    return X, Y

# -------------------------------------------------
# 3) PyTorch MLP 모델 정의 (Multi-Label Output)
# -------------------------------------------------
class LottoNet(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, output_dim=45):
        super(LottoNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # multi-label classification -> sigmoid
        x = torch.sigmoid(self.fc3(x))
        return x

# -------------------------------------------------
# 4) 훈련 함수 예시
# -------------------------------------------------
def train_model(X_train, Y_train, X_val, Y_val, 
                epochs=20, batch_size=16, lr=1e-3, 
                hidden_dim=64):
    """
    X_train, Y_train: numpy array
    X_val, Y_val    : numpy array
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LottoNet(input_dim=45, hidden_dim=hidden_dim, output_dim=45).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # 다중 라벨 -> BCELoss (출력에 sigmoid가 있으므로)
    
    # 데이터를 텐서로 변환
    X_train_t = torch.from_numpy(X_train).to(device)
    Y_train_t = torch.from_numpy(Y_train).to(device)
    X_val_t   = torch.from_numpy(X_val).to(device)
    Y_val_t   = torch.from_numpy(Y_val).to(device)
    
    train_size = X_train_t.shape[0]
    num_batches = (train_size + batch_size - 1) // batch_size
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        
        # 미니배치 학습
        perm = torch.randperm(train_size)
        for b_idx in range(num_batches):
            batch_indices = perm[b_idx*batch_size : (b_idx+1)*batch_size]
            bx = X_train_t[batch_indices]
            by = Y_train_t[batch_indices]
            
            optimizer.zero_grad()
            pred = model(bx)        # (batch, 45), sigmoid
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, train_loss={running_loss/num_batches:.4f}, val_loss={val_loss:.4f}")
    
    return model

# -------------------------------------------------
# 5) “공번호별”로 모델을 학습해보기
# -------------------------------------------------
models_by_ball = {}
lookback=5

for ball_i in range(1,6):
    sub_df = df[df['ball'] == ball_i].copy()
    if len(sub_df) < lookback+1:
        print(f"[Ball {ball_i}] 데이터가 적어서 스킵합니다.")
        continue
    
    # (a) Dataset 구성
    X, Y = build_dataset_for_ball(sub_df, lookback=lookback)
    if len(X) < 5:
        print(f"[Ball {ball_i}] 유효 샘플 수가 너무 적어 스킵합니다.")
        continue
    
    # (b) train/val 분할 (단순 예시)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    # (c) 모델 학습
    print(f"\n===== [Ball {ball_i}] 모델 학습 시작 =====")
    model = train_model(X_train, Y_train, X_val, Y_val, epochs=20, batch_size=16, lr=1e-3, hidden_dim=64)
    
    # (d) val set 정확도(혹은 F1 등) 측정 (간단히 0.5 threshold로 multi-label Accuracy 계산)
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)
    
    with torch.no_grad():
        preds = model(X_val_t)  # shape=(val_size, 45)
    preds_np = preds.cpu().numpy()
    y_val_np = Y_val_t.cpu().numpy()
    
    # multi-label 예측(>0.5면 1)
    pred_binary = (preds_np > 0.5).astype(int)
    # 정확도(각 샘플마다 45개 라벨 중 맞춘 비율)의 평균
    sample_accuracies = []
    for i in range(len(pred_binary)):
        match_count = np.sum(pred_binary[i] == y_val_np[i])
        sample_acc  = match_count / 45
        sample_accuracies.append(sample_acc)
    mean_acc = np.mean(sample_accuracies)
    print(f"[Ball {ball_i}] Validation mean accuracy: {mean_acc:.4f}")
    
    # 모델 저장
    models_by_ball[ball_i] = model

# -------------------------------------------------
# 6) 새로 들어온 "최근 5회" 데이터로 예측해보기 (예시)
# -------------------------------------------------
def predict_for_ball(ball_i, recent_rows_df, model, lookback=5, top_k=6):
    """
    recent_rows_df: 최근 5회 이상의 row들이 있는 DataFrame (공번호=ball_i)
                    round 오름차순 정렬 가정
    model: 학습된 PyTorch 모델
    return: 확률 상위 top_k개 번호 (예: 메인용)
    """
    # 1) 가중치 합 feature
    sets_of_numbers = list(recent_rows_df['numbers_set'])
    freq_vec = np.zeros(45, dtype=float)
    
    for offset in range(lookback):
        weight = lookback - offset
        row_set = sets_of_numbers[-(offset+1)]  # 마지막부터
        for num in row_set:
            freq_vec[num-1] += weight
    
    freq_t = torch.from_numpy(freq_vec).float().unsqueeze(0)
    device = next(model.parameters()).device
    freq_t = freq_t.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(freq_t)  # shape=(1,45)
    output_np = output.cpu().numpy().reshape(-1)  # 45차원
    
    # 확률 높은 순으로 정렬
    sorted_indices = np.argsort(-output_np)  # 내림차순
    top_numbers = sorted_indices[:top_k] + 1  # 0-based -> 번호(1~45)
    
    return top_numbers

# 예: 공번호=1에 대해, 가장 최근 5회 데이터로 예측
# ball_i = 1
for ball_i in range(1,6):
    if ball_i in models_by_ball:
        sub_df = df[df['ball'] == ball_i].copy().sort_values('round')
        recent_df = sub_df.tail(5)
        pred_nums = predict_for_ball(ball_i, recent_df, models_by_ball[ball_i], lookback=5, top_k=6)
        print(f"\n[Ball {ball_i}] 예측 번호(상위6): {pred_nums}")
    else:
        print(f"[Ball {ball_i}] 모델이 없습니다.")

print()