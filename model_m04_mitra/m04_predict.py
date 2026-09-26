"""
m04 주간 배치 예측
===================
최신 회차까지의 이력으로 **다음 회차** 번호별 확률표(공세트 5 × 번호 45)를 만들어 저장한다.

  1. 이력 로드 (Supabase 또는 CSV)
  2. feature 생성 → 최근 context_rounds 회차를 컨텍스트로 Mitra-v2 ICL
  3. 공세트 1~5 가정 query 225행 예측 → 회차 단위로 합 6 정규화 → posterior(합 1)
  4. prediction.json 저장 + Supabase model_predictions 업서트 (--dry-run 이면 생략)

서버는 autogluon/torch 없이 model_predictions 만 읽는다 (web/number_gen.py).

실행 (아이맥, m04 전용 venv):
  python model_m04_mitra/m04_predict.py --source supabase
  python model_m04_mitra/m04_predict.py --source csv --backend logreg --dry-run   # 파이프라인 점검
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m04_data                                           # noqa: E402
from m04_features import build, context                   # noqa: E402
from m04_model import make_backend, normalize_round, to_posterior   # noqa: E402

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_PATH  = os.path.join(HERE, "prediction.json")
MODEL_KEY = "m04"


def predict(source: str, backend: str, context_rounds: int, **backend_kw) -> dict:
    hist = m04_data.load(source)
    ft = build(hist, use_rehearsal=backend_kw.get("use_rehearsal", False))
    X, y = context(ft, len(hist), context_rounds)

    t0 = time.time()
    model = make_backend(backend, **backend_kw).fit(X, y)
    q = ft.next_query                                       # (5, 45, F)
    raw = model.predict(q.reshape(-1, q.shape[-1])).reshape(5, 45)
    probs = np.stack([to_posterior(normalize_round(r)) for r in raw])

    last = int(hist.rounds[-1])
    return {
        "model": MODEL_KEY,
        "backend": backend,
        "target_round": last + 1,
        "trained_through_round": last,
        "context_rows": int(len(y)),
        "probs": {str(b): probs[b - 1].round(6).tolist() for b in range(1, 6)},
        "config": {"context_rounds": context_rounds, "features": ft.names,
                   **{k: v for k, v in backend_kw.items() if v is not None}},
        "elapsed_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert(pred: dict):
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(os.path.join(m04_data.ROOT, ".env"))
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    sb = create_client(os.getenv("SUPABASE_URL"), key)
    rows = [
        {
            "model": pred["model"],
            "target_round": pred["target_round"],
            "ball_set": int(b),
            "probs": p,
            "trained_through_round": pred["trained_through_round"],
            "backend": pred["backend"],
            "config": pred["config"],
            "created_at": pred["created_at"],
        }
        for b, p in pred["probs"].items()
    ]
    sb.table("model_predictions").upsert(rows, on_conflict="model,target_round,ball_set").execute()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="supabase", choices=["csv", "supabase"])
    ap.add_argument("--backend", default="mitra", help="mitra | logreg | m03 | uniform")
    ap.add_argument("--context-rounds", type=int, default=110)
    ap.add_argument("--use-rehearsal", action="store_true")
    ap.add_argument("--fine-tune", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hf-model", default="autogluon/mitra-classifier-2")
    ap.add_argument("--dry-run", action="store_true", help="DB 업서트 생략")
    ap.add_argument("--force", action="store_true", help="mitra 가 아닌 백엔드도 m04 로 업서트 (디버그용)")
    args = ap.parse_args()

    pred = predict(
        args.source, args.backend, args.context_rounds,
        use_rehearsal=args.use_rehearsal, fine_tune=args.fine_tune,
        n_estimators=args.n_estimators, device=args.device, hf_model=args.hf_model,
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)

    print(f"[m04] {pred['target_round']}회차 예측 (학습 ~{pred['trained_through_round']}회, "
          f"backend={pred['backend']}, 컨텍스트 {pred['context_rows']}행, {pred['elapsed_seconds']}s)")
    for b, p in pred["probs"].items():
        top = np.argsort(p)[::-1][:6] + 1
        print(f"  공세트 {b}: 상위 6 {sorted(top.tolist())}  (max {max(p) * 100:.2f}%, min {min(p) * 100:.2f}%)")

    if args.dry_run:
        print("[m04] --dry-run: DB 업서트 생략")
        return
    if args.backend != "mitra" and not args.force:
        print(f"[m04] backend={args.backend} 는 m04 로 게시하지 않음 (--force 로 강제)")
        return
    upsert(pred)
    print("[m04] model_predictions 업서트 완료")


if __name__ == "__main__":
    main()
