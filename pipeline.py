"""
pipeline.py — 매주 금요일 자동 실행 파이프라인
================================================

실행 순서:
  1. 크롤링  : 네이버 카페에서 최신 추첨 데이터 수집 → history_from_cafe.csv 갱신
  2. 학습    : m02_train.py로 모델 재학습 → best_m02.pth 갱신
  3. 예측    : 공세트 1~5 각각의 최고 조합(전략 1 1위) 추출
  4. 구매    : 5장 수동 번호 구매 (공세트별 1장씩)
  5. 로그    : pipeline_log.txt 에 실행 결과 기록

환경 변수:
  LOTTO_USER_ID   동행복권 아이디
  LOTTO_USER_PW   동행복권 비밀번호
  SKIP_CRAWL      1 이면 크롤링 건너뜀 (디버그용)
  SKIP_TRAIN      1 이면 학습 건너뜀 (디버그용)
  SKIP_PURCHASE   1 이면 구매 건너뜀 (디버그용)

단독 실행:
  python pipeline.py

LaunchAgent 등록 (매주 금요일 오전 10시 KST):
  ~/Library/LaunchAgents/com.lotto645.pipeline.plist 참고
  launchctl load ~/Library/LaunchAgents/com.lotto645.pipeline.plist
"""

import os
import sys
import subprocess
import itertools
import json
from datetime import datetime
from pathlib import Path

import pytz
import numpy as np
import pandas as pd
import torch

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
PYTHON      = sys.executable
CRAWL_SCRIPT = ROOT / "crawling/01_dh_caffe_crawling_with_auto_login.py"
TRAIN_SCRIPT = ROOT / "model_m02_claude/m02_train.py"
DATA_PATH    = ROOT / "data/history_from_cafe.csv"
CKPT_PATH    = ROOT / "model_m02_claude/best_m02.pth"
LOG_PATH     = ROOT / "pipeline_log.txt"
RESULT_PATH  = ROOT / "pipeline_result.json"

NUM_BALLS = 45
NUM_SETS  = 5
TOP_N     = 15   # 번호 풀 크기


# ── 로그 헬퍼 ─────────────────────────────────────────────────────────────────
def log(msg: str):
    kst  = pytz.timezone("Asia/Seoul")
    ts   = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


# ── Step 1: 크롤링 ────────────────────────────────────────────────────────────
def step_crawl() -> bool:
    if os.getenv("SKIP_CRAWL") == "1":
        log("SKIP_CRAWL=1 → 크롤링 건너뜀")
        return True
    log("Step 1: 크롤링 시작")
    result = subprocess.run(
        [PYTHON, str(CRAWL_SCRIPT)],
        cwd=str(ROOT / "crawling"),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        log(f"크롤링 실패 (returncode={result.returncode})")
        return False
    log("크롤링 완료")
    return True


# ── Step 2: 학습 ──────────────────────────────────────────────────────────────
def step_train() -> bool:
    if os.getenv("SKIP_TRAIN") == "1":
        log("SKIP_TRAIN=1 → 학습 건너뜀")
        return True
    log("Step 2: 모델 학습 시작")
    result = subprocess.run(
        [PYTHON, str(TRAIN_SCRIPT)],
        cwd=str(ROOT / "model_m02_claude"),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        log(f"학습 실패 (returncode={result.returncode})")
        return False
    log("학습 완료")
    return True


# ── Step 3: 예측 (공세트 1~5 최고 조합) ──────────────────────────────────────
def step_predict() -> list[list[int]] | None:
    log("Step 3: 공세트별 번호 예측 시작")

    # 모델/데이터 로드
    sys.path.insert(0, str(ROOT / "model_m02_claude"))
    from m02_model import BallSetLSTM

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt  = torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    cfg   = ckpt["config"]
    model = BallSetLSTM(
        num_balls=cfg["num_balls"], num_ball_sets=cfg["num_ball_sets"],
        emb_dim=cfg["emb_dim"], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(str(DATA_PATH), header=None,
                     names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"])
    df = df.sort_values(["round","ball_set"]).reset_index(drop=True)

    win_size = cfg["win_size"]
    tickets  = []   # 공세트 1~5 각 최고 조합

    for bs in range(1, NUM_SETS + 1):
        # 확률 계산
        rows = df.tail(win_size).reset_index(drop=True)
        pad  = win_size - len(rows)
        oh_list, bs_list = [], []
        for _, row in rows.iterrows():
            arr = np.zeros(NUM_BALLS, dtype=np.float32)
            for c in ["n1","n2","n3","n4","n5","n6","bonus"]:
                v = int(row[c])
                if 1 <= v <= NUM_BALLS:
                    arr[v-1] = 1.0
            oh_list.append(arr)
            bs_list.append(int(row["ball_set"]) - 1)
        if pad > 0:
            oh_list = [np.zeros(NUM_BALLS, dtype=np.float32)] * pad + oh_list
            bs_list = [0] * pad + bs_list

        x_oh = torch.from_numpy(np.array(oh_list)).unsqueeze(0).to(device)
        x_bs = torch.tensor([bs_list], dtype=torch.long).to(device)
        t_bs = torch.tensor([bs - 1], dtype=torch.long).to(device)
        with torch.no_grad():
            prob = model(x_oh, x_bs, t_bs)[0].cpu().numpy()

        # 상위 TOP_N 번호 풀에서 최고 조합(전략 1, 1위)
        top_idx  = np.argsort(prob)[::-1][:TOP_N]
        top_nums = sorted((top_idx + 1).tolist())

        best_combo = None
        best_score = -1
        for combo in itertools.combinations(top_nums, 6):
            sc = sum(prob[n-1] for n in combo)
            if sc > best_score:
                best_score = sc
                best_combo = list(combo)

        log(f"  공세트 {bs}: {best_combo} (score={best_score*100:.2f}%)")
        tickets.append(best_combo)

    # 결과 JSON 저장
    result = {
        "predicted_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
        "tickets": {f"ball_set_{i+1}": t for i, t in enumerate(tickets)},
    }
    with open(str(RESULT_PATH), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"예측 완료. 결과 저장: {RESULT_PATH}")
    return tickets


# ── Step 4: 구매 ──────────────────────────────────────────────────────────────
def step_purchase(tickets: list[list[int]]) -> bool:
    if os.getenv("SKIP_PURCHASE") == "1":
        log("SKIP_PURCHASE=1 → 구매 건너뜀")
        log(f"  (구매 예정 번호: {tickets})")
        return True

    log("Step 4: 자동 구매 시작")
    sys.path.insert(0, str(ROOT / "purchase"))
    from purchase_with_numbers import buy_with_numbers

    success = buy_with_numbers(tickets)
    if success:
        log("구매 완료")
    else:
        log("구매 실패 — purchase_error.png / purchase_failure.png 확인")
    return success


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────
def main():
    kst = pytz.timezone("Asia/Seoul")
    log("=" * 60)
    log("파이프라인 시작")
    log("=" * 60)

    # 1. 크롤링
    if not step_crawl():
        log("[중단] 크롤링 실패")
        sys.exit(1)

    # 2. 학습
    if not step_train():
        log("[중단] 학습 실패")
        sys.exit(1)

    # 3. 예측
    tickets = step_predict()
    if tickets is None:
        log("[중단] 예측 실패")
        sys.exit(1)

    # 4. 구매
    step_purchase(tickets)

    log("=" * 60)
    log("파이프라인 완료")
    log("=" * 60)


if __name__ == "__main__":
    main()
