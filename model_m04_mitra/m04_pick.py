"""
m04 티켓 선택 — 확률(m04) × 인기 회피
======================================
목표: 1등 **기대 가치**가 가장 큰 조합 N장.

  log 기대가치(c) = Σ_i log(p_i / (6/45))        ← 모델이 본 당첨 확률 (균등 대비 배수)
                  − log 공동당첨배수(c)            ← 남들이 덜 고를수록 덜 나눔 (m04_popularity)

두 항이 같은 단위(로그 배수)라 임의 가중치가 없다. 모델 확률이 거의 균등하면 인기 회피가 주도하고,
모델에 신호가 있으면 그만큼 반영된다. `--weight` 로 인기 항 비중만 조절할 수 있다(기본 1).

공세트는 구매 시점에 모르므로 기본은 과거 공세트 출현 빈도로 5개 세트 확률을 섞는다(--ball-set mix).
과거 1등 당첨 조합은 제외한다 (그대로 다시 사는 사람이 많아 인기 조합).

사용:
  python m04_pick.py                                   # prediction.json(m04_predict 결과) 사용
  python m04_pick.py --backend m03r --use-rehearsal --alpha 100   # 그 자리에서 확률 계산
  python m04_pick.py --n 5 --max-overlap 2 --ball-set mix
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m04_data                                   # noqa: E402
from m04_data import NUM_BALLS, NUM_SETS          # noqa: E402
from m04_popularity import fit as fit_popularity  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_RATE = 6 / NUM_BALLS


def set_weights(hist) -> np.ndarray:
    """과거 본게임의 공세트 출현 빈도 (1~5)"""
    cnt = np.array([(hist.ball_set == b).sum() for b in range(1, NUM_SETS + 1)], dtype=float)
    return cnt / cnt.sum()


def mixed_rates(probs: dict, hist, ball_set: str) -> np.ndarray:
    """probs: {'1': [45 posterior], ...} (합 1) → 번호별 출현 확률 (합 6)"""
    table = np.array([probs[str(b)] for b in range(1, NUM_SETS + 1)], dtype=float)
    table = table / table.sum(axis=1, keepdims=True) * 6
    if ball_set == "mix":
        return set_weights(hist) @ table
    return table[int(ball_set) - 1]


def sample_combos(n: int, rng) -> np.ndarray:
    """서로 다른 6개 번호 조합 n개 (정렬, 1-based, 중복 제거)"""
    c = np.sort(np.argsort(rng.random((n, NUM_BALLS)), axis=1)[:, :6] + 1, axis=1)
    return np.unique(c, axis=0)


def pick(rates: np.ndarray, pm, hist, n: int = 5, max_overlap: int = 2, weight: float = 1.0,
         samples: int = 400_000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    cands = sample_combos(samples, rng)

    past = {tuple(np.flatnonzero(h) + 1) for h in hist.hits}
    keep = np.array([tuple(c) not in past for c in cands])
    cands = cands[keep]

    log_p = np.log(np.clip(rates, 1e-6, None) / BASE_RATE)
    model_term = log_p[cands - 1].sum(axis=1)
    share = pm.jackpot_share(cands)
    objective = model_term - weight * np.log(share)

    chosen = []
    for i in np.argsort(-objective):
        c = set(cands[i].tolist())
        if all(len(c & set(t["numbers"])) <= max_overlap for t in chosen):
            chosen.append({
                "numbers": cands[i].tolist(),
                "ev_multiplier": float(np.exp(objective[i])),        # 1등 기대가치, 평균 조합 대비
                "prob_multiplier": float(np.exp(model_term[i])),     # 모델이 본 당첨확률, 균등 대비
                "co_winner_multiplier": float(share[i]),             # 공동당첨자 수, 평균 조합 대비
                "popularity_percentile": float((share < share[i]).mean() * 100),
            })
            if len(chosen) == n:
                break
    return chosen


def load_probs(args, hist) -> tuple[dict, dict]:
    if args.backend:
        from m04_predict import predict
        pred = predict(args.source, args.backend, args.context_rounds,
                       use_rehearsal=args.use_rehearsal, alpha=args.alpha, device=args.device)
    else:
        path = args.prediction or os.path.join(HERE, "prediction.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path} 없음 — 먼저 m04_predict.py 를 돌리거나 --backend 를 지정하세요")
        with open(path, encoding="utf-8") as f:
            pred = json.load(f)
    return pred["probs"], {k: pred.get(k) for k in ("backend", "target_round", "trained_through_round", "config")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="csv", choices=["csv", "supabase"])
    ap.add_argument("--prediction", default=None, help="m04_predict.py 결과 JSON (기본 prediction.json)")
    ap.add_argument("--backend", default=None, help="지정하면 prediction.json 대신 그 자리에서 확률 계산")
    ap.add_argument("--use-rehearsal", action="store_true")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--context-rounds", type=int, default=110)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ball-set", default="mix", help="mix(기본, 과거 빈도로 섞음) 또는 1~5")
    ap.add_argument("--n", type=int, default=5, help="티켓 수")
    ap.add_argument("--max-overlap", type=int, default=2, help="티켓끼리 겹쳐도 되는 번호 수")
    ap.add_argument("--weight", type=float, default=1.0, help="인기 회피 비중 (0 이면 확률만)")
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "picks.json"))
    args = ap.parse_args()

    hist = m04_data.load(args.source)
    probs, meta = load_probs(args, hist)
    rates = mixed_rates(probs, hist, args.ball_set)
    pm = fit_popularity(hist)
    tickets = pick(rates, pm, hist, n=args.n, max_overlap=args.max_overlap,
                   weight=args.weight, samples=args.samples, seed=args.seed)

    target = meta.get("target_round") or int(hist.rounds[-1]) + 1
    print(f"[m04 pick] {target}회 | 확률: {meta.get('backend')} (데이터 ~{meta.get('trained_through_round')}회) "
          f"| 공세트 {args.ball_set} | 인기 모델 γ={pm.gamma:.2f}±{pm.gamma_se:.2f}\n")
    print("| # | 번호 | 1등 기대가치 | 당첨확률(모델) | 공동당첨 | 더 비인기인 조합 |")
    print("|---|---|---|---|---|---|")
    for i, t in enumerate(tickets, 1):
        print(f"| {i} | {' · '.join(map(str, t['numbers']))} | x{t['ev_multiplier']:.2f} | "
              f"x{t['prob_multiplier']:.3f} | x{t['co_winner_multiplier']:.2f} | {t['popularity_percentile']:.2f}% |")
    print("\n기대가치·공동당첨은 평균 조합 대비. '더 비인기인 조합' = 전체 조합 중 이 조합보다 덜 인기인 비율.")
    print("당첨확률 배수는 모델 추정치이며 백테스트로 확정된 값이 아님. 공동당첨 배수는 과거 당첨조합 범위 밖으로")
    print("외삽된 값일 수 있음(극단적 비인기 조합은 과거 당첨번호에 드묾).")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target_round": target, "meta": meta, "ball_set": args.ball_set,
                   "popularity": {"gamma": pm.gamma, "gamma_se": pm.gamma_se, "n_rounds": pm.n_rounds},
                   "tickets": tickets}, f, ensure_ascii=False, indent=2)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
