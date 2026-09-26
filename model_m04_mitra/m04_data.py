"""
m04 데이터 로더
================
추첨 이력을 회차 순서의 배열 묶음(History)으로 바꾼다.

원천
  · CSV  : data/history_from_cafe.csv (회차당 1행이면 실제 당첨번호, 2행이면 첫 행=리허설 / 둘째 행=실제)
  · DB   : Supabase draw_results (is_winning 컬럼으로 구분)

m03 의 fetch_history() 와 달리 ball_set 이 없는 초기 회차(733~834)도 버리지 않는다.
전체 빈도/간격 feature 는 이 회차들까지 누적해서 계산하고, 학습 행(컨텍스트)은 ball_set 이 있는 회차만 쓴다.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "history_from_cafe.csv")

NUM_BALLS = 45
NUM_SETS  = 5
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


@dataclass
class History:
    """회차 오름차순. 길이 R."""
    rounds:   np.ndarray   # (R,) int
    ball_set: np.ndarray   # (R,) int, 0 = 모름
    hits:     np.ndarray   # (R, 45) bool — 본번호 6개
    bonus:    np.ndarray   # (R,) int
    reh_hits: np.ndarray   # (R, 45) bool — 같은 회차 리허설 본번호 (없으면 전부 False)
    has_reh:  np.ndarray   # (R,) bool

    def __len__(self) -> int:
        return len(self.rounds)

    def head(self, n: int) -> "History":
        """앞에서 n 회차만, 복사본으로 (백테스트/누수 테스트용)"""
        return History(*(getattr(self, f)[:n].copy() for f in self.__dataclass_fields__))


def _to_hits(nums: pd.DataFrame) -> np.ndarray:
    arr = np.zeros((len(nums), NUM_BALLS), dtype=bool)
    vals = nums.to_numpy(dtype=int)
    for i, row in enumerate(vals):
        arr[i, row - 1] = True
    return arr


def from_frame(df: pd.DataFrame) -> History:
    """
    df 컬럼: round, is_winning, ball_set(NaN 가능), n1~n6, bonus
    """
    df = df.copy()
    df["ball_set"] = pd.to_numeric(df["ball_set"], errors="coerce").fillna(0).astype(int)
    real = df[df["is_winning"].astype(bool)].sort_values("round").drop_duplicates("round", keep="last")
    reh  = df[~df["is_winning"].astype(bool)].drop_duplicates("round", keep="last").set_index("round")

    rounds = real["round"].to_numpy(dtype=int)
    reh_hits = np.zeros((len(real), NUM_BALLS), dtype=bool)
    has_reh  = np.zeros(len(real), dtype=bool)
    if len(reh):
        reh_all = _to_hits(reh[MAIN_COLS])
        pos = {r: i for i, r in enumerate(reh.index.to_numpy(dtype=int))}
        for i, r in enumerate(rounds):
            if r in pos:
                reh_hits[i] = reh_all[pos[r]]
                has_reh[i]  = True

    return History(
        rounds=rounds,
        ball_set=real["ball_set"].to_numpy(dtype=int),
        hits=_to_hits(real[MAIN_COLS]),
        bonus=real["bonus"].to_numpy(dtype=int),
        reh_hits=reh_hits,
        has_reh=has_reh,
    )


def load_csv(path: str = CSV_PATH) -> History:
    df = pd.read_csv(path, header=None, usecols=range(10),
                     names=["ball_set", "round", "draw_date", *MAIN_COLS, "bonus"])
    # 회차당 1행 = 실제, 2행 = (리허설, 실제)  — scripts/backfill_csv_from_db.py 와 같은 규칙
    df["is_winning"] = (df.groupby("round")["round"].transform("size") == 1) | (df.groupby("round").cumcount() > 0)
    return from_frame(df)


def load_supabase() -> History:
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(os.path.join(ROOT, ".env"))
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    def page(start: int, end: int):
        # postgrest range() 는 offset/limit 을 덮어쓰지 않고 누적하므로 페이지마다 쿼리를 새로 만든다
        return (
            sb.table("draw_results")
            .select("round,is_winning,ball_set," + ",".join(MAIN_COLS) + ",bonus")
            .order("round")
            .order("is_winning")
            .range(start, end)
            .execute()
            .data
        )

    rows, start, step = [], 0, 1000
    while True:
        r = page(start, start + step - 1)
        if not r:
            break
        rows.extend(r)
        if len(r) < step:
            break
        start += step
    return from_frame(pd.DataFrame(rows))


def load(source: str = "csv") -> History:
    if source == "csv":
        return load_csv()
    if source == "supabase":
        return load_supabase()
    raise ValueError(f"source 는 csv|supabase: {source}")
