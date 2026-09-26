"""
m04 백테스트 (rolling-origin)
==============================
회차 t 마다: t 이전 데이터만으로 컨텍스트 구성 → t 의 실제 공세트 기준 45개 번호 확률 예측 → 채점.

평가 구간
  · 기본(dev) : 공세트가 있는 회차 중 [끝-holdout-n_rounds, 끝-holdout)
  · --final   : 마지막 holdout 회차. 설정(feature/컨텍스트/백엔드 옵션)을 확정한 뒤 **한 번만** 돌린다.

지표 (균등분포 기대값)
  · logloss    : 번호별 이진 log loss (≈0.3927)          낮을수록 좋음
  · brier      : (≈0.1156)                               낮을수록 좋음
  · top15_hits : 확률 상위 15개 중 당첨 수 (2.0)          높을수록 좋음 — 서비스 번호 풀
  · top6_hits  : 확률 상위 6개(=전략 1 티켓) 일치 수 (0.8)  높을수록 좋음

비교: 회차별 (백엔드 − 기준) 차이의 paired bootstrap 95% CI. CI 가 0 을 넘어야 "개선".

사용 예
  python m04_backtest.py --backend uniform m03 logreg
  python m04_backtest.py --backend uniform m03 mitra --n-rounds 100
  python m04_backtest.py --backend mitra --context same_set --use-rehearsal
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m04_data                                                  # noqa: E402
from m04_features import build, context, context_same_set        # noqa: E402
from m04_model import make_backend, normalize_round              # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest")
METRICS = {"logloss": -1, "brier": -1, "top15_hits": +1, "top6_hits": +1}   # +1 = 높을수록 좋음


def score(p: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    br = ((p - y) ** 2).mean()
    order = np.lexsort((rng.random(len(p)), -p))   # 동점은 무작위 (uniform 이 번호 순으로 뽑히지 않게)
    return {
        "logloss": float(ll),
        "brier": float(br),
        "top15_hits": int(y[order[:15]].sum()),
        "top6_hits": int(y[order[:6]].sum()),
    }


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(args) -> dict:
    hist = m04_data.load(args.source)
    ft = build(hist, use_rehearsal=args.use_rehearsal)
    evaluable = [k for k in range(len(hist)) if ft.per_round[k] is not None]
    if args.final:
        targets = evaluable[-args.holdout:]
    else:
        end = len(evaluable) - args.holdout
        targets = evaluable[max(0, end - args.n_rounds):end]

    backend_kw = dict(
        use_rehearsal=args.use_rehearsal,
        fine_tune=args.fine_tune,
        n_estimators=args.n_estimators,
        device=args.device,
        hf_model=args.hf_model,
    )
    rng = np.random.default_rng(args.seed)
    per_round = {b: [] for b in args.backend}
    t0 = time.time()
    for i, k in enumerate(targets):
        b = int(ft.ball_set[k])
        if args.context == "same_set":
            X, y = context_same_set(ft, k, args.context_rounds, b)
        else:
            X, y = context(ft, k, args.context_rounds)
        for name in args.backend:
            model = make_backend(name, **backend_kw).fit(X, y)
            p = normalize_round(model.predict(ft.per_round[k]))
            per_round[name].append(score(p, ft.labels[k], rng))
        if (i + 1) % 10 == 0 or i + 1 == len(targets):
            print(f"  {i + 1}/{len(targets)} 회차  ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for name, rows in per_round.items():
        summary[name] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}

    comparisons = {}
    for ref in ("uniform", "m03"):
        if ref not in per_round:
            continue
        for name in args.backend:
            if name == ref or f"{ref}_vs_{name}" in comparisons:
                continue
            comp = {}
            for m, sign in METRICS.items():
                diff = sign * (np.array([r[m] for r in per_round[name]]) - np.array([r[m] for r in per_round[ref]]))
                lo, hi = bootstrap_ci(diff, rng)
                comp[m] = {"mean_gain": float(diff.mean()), "ci95": [lo, hi], "better": lo > 0}
            comparisons[f"{name}_vs_{ref}"] = comp

    return {
        "config": {k: v for k, v in vars(args).items()},
        "rounds": [int(ft.rounds[targets[0]]), int(ft.rounds[targets[-1]])],
        "n_rounds": len(targets),
        "context_rows": int(len(y)),
        "summary": summary,
        "comparisons": comparisons,
    }


def print_report(res: dict):
    print(f"\n평가 회차 {res['rounds'][0]}~{res['rounds'][1]} ({res['n_rounds']}회), 컨텍스트 {res['context_rows']}행")
    print(f"\n| backend | logloss ↓ | brier ↓ | top15 적중 ↑ | 전략1 일치 ↑ |")
    print(f"|---|---|---|---|---|")
    print(f"| (균등 기대값) | 0.3927 | 0.1156 | 2.000 | 0.800 |")
    for name, s in res["summary"].items():
        print(f"| {name} | {s['logloss']:.4f} | {s['brier']:.4f} | {s['top15_hits']:.3f} | {s['top6_hits']:.3f} |")
    if res["comparisons"]:
        print("\n개선폭 (+ 가 좋음) 과 95% CI — CI 전체가 0 초과일 때만 '개선'")
        for key, comp in res["comparisons"].items():
            cells = [f"{m} {c['mean_gain']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]{' ✅' if c['better'] else ''}"
                     for m, c in comp.items()]
            print(f"  {key}: " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="csv", choices=["csv", "supabase"])
    ap.add_argument("--backend", nargs="+", default=["uniform", "m03", "logreg"])
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=30, help="최종 평가용으로 떼어둘 최근 회차 수")
    ap.add_argument("--final", action="store_true", help="holdout 구간 평가 (설정 확정 후 1회만)")
    ap.add_argument("--context", default="recent", choices=["recent", "same_set"])
    ap.add_argument("--context-rounds", type=int, default=110, help="컨텍스트 회차 수 (×45행, Mitra-v2 한도 5,120행)")
    ap.add_argument("--use-rehearsal", action="store_true", help="이전 회차 리허설 기반 feature 추가")
    ap.add_argument("--fine-tune", action="store_true", help="Mitra fine-tune (GPU 권장)")
    ap.add_argument("--n-estimators", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hf-model", default="autogluon/mitra-classifier-2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (기본: backtest/ 아래 타임스탬프)")
    args = ap.parse_args()

    if args.context_rounds * 45 > 5120:
        print(f"⚠️  컨텍스트 {args.context_rounds * 45}행 — Mitra-v2 사전학습 컨텍스트(5,120행)를 넘습니다")

    res = run(args)
    print_report(res)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"backtest_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
