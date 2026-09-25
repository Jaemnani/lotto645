"""공세트별 번호 출현이 균등(각 6/45)한지 검정. 본게임 / 리허설 / 합산 + 리허설-본게임 재현성.

실행: python scripts/ballset_uniformity.py   (numpy 필요, 약 1분)
"""
import csv
import os
import collections
import numpy as np

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "history_from_cafe.csv")
N_SIM = 20000
rng = np.random.default_rng(0)

by_round = collections.defaultdict(list)
for r in csv.reader(open(CSV)):
    by_round[int(r[1])].append(r)

draws = []  # (round, ball_set, kind, [6 numbers])
for rnd, rows in sorted(by_round.items()):
    if len(rows) != 2 or not rows[1][0]:
        continue
    bs = int(rows[1][0])
    assert rows[0][0] == rows[1][0]
    draws.append((rnd, bs, "rehearsal", [int(x) for x in rows[0][3:9]]))
    draws.append((rnd, bs, "real", [int(x) for x in rows[1][3:9]]))


def counts_of(ds):
    c = np.zeros(45)
    for d in ds:
        for n in d[3]:
            c[n - 1] += 1
    return c


def chi2(c, n_draws):
    e = n_draws * 6 / 45
    return ((c - e) ** 2 / e).sum()


def sim_null(n_draws):
    """균등 무작위 추첨(6개 비복원)을 n_draws회 했을 때 chi2 분포"""
    c = np.zeros((N_SIM, 45))
    B = 500
    for s0 in range(0, N_SIM, B):
        picks = np.argpartition(rng.random((B, n_draws, 45)), 6, axis=2)[:, :, :6]
        flat = picks.reshape(B, -1) + (np.arange(B) * 45)[:, None]
        c[s0:s0 + B] = np.bincount(flat.ravel(), minlength=B * 45).reshape(B, 45)
    e = n_draws * 6 / 45
    return ((c - e) ** 2 / e).sum(axis=1), c


print(f"회차 {draws[0][0]}~{draws[-1][0]}, 공세트 있는 회차 {len(draws)//2}개\n")
print("=== 1) 공세트별 균등성 검정 (Monte Carlo chi-square, 각 번호 기대 13.33%) ===")
print(f"{'세트':>4} {'구분':<10} {'추첨수':>5} {'최저번호(%)':>14} {'최고번호(%)':>14} {'p-value':>8}")
summary = {}
for bs in range(1, 6):
    for kind in ["real", "rehearsal", "both"]:
        ds = [d for d in draws if d[1] == bs and (kind == "both" or d[2] == kind)]
        n = len(ds)
        c = counts_of(ds)
        stat = chi2(c, n)
        null, null_c = sim_null(n)
        p = (null >= stat).mean()
        rate = c / n * 100
        lo, hi = rate.argmin(), rate.argmax()
        print(f"{bs:>4} {kind:<10} {n:>5} {lo+1:>4}번 {rate[lo]:5.1f}%   {hi+1:>4}번 {rate[hi]:5.1f}%   {p:8.3f}")
        if kind == "both":
            # 무작위라도 최고/최저 번호는 이 정도로 벌어진다 (참고용)
            sim_rates = null_c / n * 100
            summary[bs] = (n, np.percentile(sim_rates.max(axis=1), 50), np.percentile(sim_rates.min(axis=1), 50))

print("\n참고: 완전 무작위여도 45개 중 최고/최저 번호의 출현율 (합산 기준, 중앙값)")
for bs, (n, mx, mn) in summary.items():
    print(f"  세트 {bs} ({n}회): 최고 ≈ {mx:.1f}%, 최저 ≈ {mn:.1f}%")

print("\n=== 2) 세트 간 차이 검정 (세트 라벨 섞기 permutation) — m03 가설 직접 검정 ===")
for kind in ["real", "both"]:
    ds = [d for d in draws if kind == "both" or d[2] == kind]
    labels = np.array([d[1] for d in ds])
    onehot = np.zeros((len(ds), 45))
    for i, d in enumerate(ds):
        onehot[i, np.array(d[3]) - 1] = 1

    def het(lbl):
        tot = onehot.sum(0) / len(ds)
        s = 0.0
        for bs in range(1, 6):
            m = lbl == bs
            e = m.sum() * tot
            s += (((onehot[m].sum(0) - e) ** 2) / e).sum()
        return s

    obs = het(labels)
    perm = np.array([het(rng.permutation(labels)) for _ in range(5000)])
    print(f"  {kind:<5}: p-value = {(perm >= obs).mean():.3f}")

print("\n=== 3) 같은 날 리허설 → 본게임 겹치는 개수 (무작위 기대 0.80개) ===")
ov = []
for i in range(0, len(draws), 2):
    ov.append(len(set(draws[i][3]) & set(draws[i + 1][3])))
ov = np.array(ov)
null_ov = np.array([len(set(rng.choice(45, 6, replace=False)) & set(range(6))) for _ in range(200000)])
print(f"  평균 {ov.mean():.3f}개 (n={len(ov)}), 분포 {collections.Counter(ov.tolist())}")
sim_means = rng.choice(null_ov, size=(20000, len(ov))).mean(axis=1)
print(f"  무작위 기준 p-value(양측) = {(np.abs(sim_means - 0.8) >= abs(ov.mean() - 0.8)).mean():.3f}")

print("\n=== 4) 이 데이터로 잡아낼 수 있는 편향 크기 (합산, 세트당 ~160회 기준) ===")
n = 160
se = np.sqrt((6 / 45) * (39 / 45) / n) * 100
print(f"  번호 하나 출현율의 표준오차 ≈ ±{se:.1f}%p → 13.3% vs {13.33 + 2.8*se:.1f}% 정도는 돼야 구분 가능 (검정력 80%, 개별 번호)")

print("\n=== 5) 재현성 검정: 같은 공으로 뽑은 리허설과 본게임의 번호별 빈도 상관 ===")
print("  (공 편향이 진짜면 본게임에서 잘 나온 번호가 리허설에서도 잘 나와야 함 → 상관 > 0)")
obs_all, per_set = [], {}
real_c = {bs: counts_of([d for d in draws if d[1] == bs and d[2] == "real"]) for bs in range(1, 6)}
reh_c = {bs: counts_of([d for d in draws if d[1] == bs and d[2] == "rehearsal"]) for bs in range(1, 6)}
for bs in range(1, 6):
    r = np.corrcoef(real_c[bs], reh_c[bs])[0, 1]
    perm = np.array([np.corrcoef(real_c[bs], rng.permutation(reh_c[bs]))[0, 1] for _ in range(20000)])
    per_set[bs] = r
    print(f"  세트 {bs}: r = {r:+.3f}   p(한쪽, r>0) = {(perm >= r).mean():.3f}")
mean_r = np.mean(list(per_set.values()))
perm_mean = np.array([
    np.mean([np.corrcoef(real_c[bs], rng.permutation(reh_c[bs]))[0, 1] for bs in range(1, 6)])
    for _ in range(5000)
])
print(f"  5개 세트 평균 r = {mean_r:+.3f}   p(한쪽) = {(perm_mean >= mean_r).mean():.3f}")
