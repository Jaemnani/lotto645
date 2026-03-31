"""
Step 2 - 학습
데이터: history_from_cafe.csv (공세트, 회차, n1~n6, bonus)
모델: BallSetLSTM (Shared LSTM + Ball-Set Conditional Head)

학습 전략
  · 슬라이딩 윈도우 방식 시퀀스 구성
  · 주번호 6개 + 보너스 1개를 multi-label (BCE) 로 예측
  · 공세트별 가중치 없이 전체 데이터 공유 학습
  · 검증: 마지막 20%를 시간 순으로 홀드아웃
  · 체크포인트: val_loss 기준 best 모델 저장
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from m02_model import BallSetLSTM, count_hits

# ── 설정 ───────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "best_m02.pth")
META_PATH  = os.path.join(os.path.dirname(__file__), "training_meta.json")

WIN_SIZE   = 32    # 입력 시퀀스 길이 (타임스텝)
NUM_BALLS  = 45
NUM_SETS   = 5     # 공세트 1~5 → 인덱스 0~4
HIDDEN     = 128
LAYERS     = 2
EMB_DIM    = 16
DROPOUT    = 0.2
BATCH_SIZE = 32
LR         = 3e-4
EPOCHS     = 200
PATIENCE   = 30    # early stopping patience


# ── 데이터셋 ───────────────────────────────────────────────────────────────────
class LottoDataset(Dataset):
    """
    슬라이딩 윈도우 데이터셋
    각 샘플:
      x_onehot     : (WIN_SIZE, 45)  과거 WIN_SIZE 회 번호 원-핫
      x_ball_sets  : (WIN_SIZE,)     과거 WIN_SIZE 회 공세트 인덱스
      target_ball  : ()              다음 회 공세트 인덱스
      y_onehot     : (45,)           다음 회 번호 원-핫 (6+보너스 = 7개)
    """

    def __init__(self, rows: pd.DataFrame, win_size: int = WIN_SIZE):
        self.win_size = win_size
        self.data = rows.reset_index(drop=True)

        # 원-핫 행렬 (N, 45)
        N = len(self.data)
        self.onehots = np.zeros((N, NUM_BALLS), dtype=np.float32)
        for i, row in self.data.iterrows():
            for col in ["n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
                v = int(row[col])
                if 1 <= v <= 45:
                    self.onehots[i, v - 1] = 1.0

        # 공세트 인덱스 (0-indexed)
        self.ball_sets = (self.data["ball_set"].values - 1).astype(np.int64)

    def __len__(self):
        return len(self.data) - self.win_size

    def __getitem__(self, idx):
        x_onehot    = self.onehots[idx: idx + self.win_size]        # (W, 45)
        x_ball_sets = self.ball_sets[idx: idx + self.win_size]      # (W,)
        target_ball = self.ball_sets[idx + self.win_size]           # scalar
        y_onehot    = self.onehots[idx + self.win_size]             # (45,)
        return (
            torch.from_numpy(x_onehot),
            torch.from_numpy(x_ball_sets),
            torch.tensor(target_ball, dtype=torch.long),
            torch.from_numpy(y_onehot),
        )


# ── 메인 학습 루프 ──────────────────────────────────────────────────────────────
def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, header=None,
                     names=["ball_set", "round", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
    df = df.sort_values(["round", "ball_set"]).reset_index(drop=True)
    print(f"데이터 로드: {len(df)}행  |  회차 {df['round'].min()}~{df['round'].max()}")
    print(f"공세트 분포:\n{df['ball_set'].value_counts().sort_index()}\n")

    # 공세트 유효성 체크
    assert df["ball_set"].between(1, NUM_SETS).all(), "공세트 번호 1~5 범위 초과"

    # ── Train/Val 분할 (시간 순) ──────────────────────────────────────────────
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    val_df   = df.iloc[split - WIN_SIZE:]   # 윈도우 overlap 보정

    train_dst = LottoDataset(train_df)
    val_dst   = LottoDataset(val_df)

    train_loader = DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_dst,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train 샘플: {len(train_dst)}, Val 샘플: {len(val_dst)}")

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = BallSetLSTM(
        num_balls=NUM_BALLS,
        num_ball_sets=NUM_SETS,
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_cnt  = 0

    # ── 학습 ──────────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_oh, x_bs, t_bs, y_oh in train_loader:
            x_oh, x_bs, t_bs, y_oh = (
                x_oh.to(device), x_bs.to(device),
                t_bs.to(device), y_oh.to(device),
            )
            optimizer.zero_grad()
            pred = model(x_oh, x_bs, t_bs)
            loss = criterion(pred, y_oh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_oh)

        train_loss /= len(train_dst)

        # ── 검증 ──────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_hits = 0.0
        n_val = 0
        with torch.no_grad():
            for x_oh, x_bs, t_bs, y_oh in val_loader:
                x_oh, x_bs, t_bs, y_oh = (
                    x_oh.to(device), x_bs.to(device),
                    t_bs.to(device), y_oh.to(device),
                )
                pred = model(x_oh, x_bs, t_bs)
                val_loss += criterion(pred, y_oh).item() * len(y_oh)

                # 상위 6개 번호 일치 수
                top6 = (pred.argsort(dim=1, descending=True)[:, :6] + 1)  # 1-indexed
                val_hits += count_hits(top6.cpu(), y_oh.cpu()) * len(y_oh)
                n_val += len(y_oh)

        val_loss /= n_val
        val_hits /= n_val

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  "
                  f"avg_hits={val_hits:.3f}/6")

        # ── 체크포인트 ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "win_size": WIN_SIZE,
                    "num_balls": NUM_BALLS,
                    "num_ball_sets": NUM_SETS,
                    "hidden_size": HIDDEN,
                    "num_layers": LAYERS,
                    "emb_dim": EMB_DIM,
                    "dropout": DROPOUT,
                },
            }, CKPT_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n학습 완료. Best val_loss={best_val_loss:.5f}")
    print(f"체크포인트 저장: {CKPT_PATH}")

    # ── 학습 메타데이터 저장 ───────────────────────────────────────────────────
    latest_round = int(df["round"].max())
    meta = {
        "last_trained_round": latest_round,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"학습 메타 저장: {META_PATH}  (last_trained_round={latest_round})")


if __name__ == "__main__":
    main()
