"""
Supabase 연결 (API Key 방식)
환경변수 SUPABASE_URL, SUPABASE_KEY 설정 필요
"""

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_client: Optional[Client] = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# FastAPI Depends 호환 이름
def get_db() -> Client:
    return get_supabase()


def init_db():
    """Supabase에서 테이블은 대시보드/마이그레이션으로 생성합니다. (no-op)"""
    pass


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class DrawResult:
    round: int
    draw_date: date
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    bonus: int
    is_winning: bool = False             # False=모의추첨, True=실제 당첨번호
    id: Optional[int] = None
    ball_set: Optional[int] = None       # 카페 갱신 전 NULL
    prize_1: Optional[int] = None        # 1등 당첨금 (원)
    prize_2: Optional[int] = None
    prize_3: Optional[int] = None
    prize_4: Optional[int] = None        # 고정 50,000
    prize_5: Optional[int] = None        # 고정 5,000
    fetched_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DrawResult":
        draw_date = d["draw_date"]
        if isinstance(draw_date, str):
            draw_date = date.fromisoformat(draw_date)
        return cls(
            id=d.get("id"),
            round=d["round"],
            draw_date=draw_date,
            is_winning=d.get("is_winning", False),
            ball_set=d.get("ball_set"),
            n1=d["n1"], n2=d["n2"], n3=d["n3"],
            n4=d["n4"], n5=d["n5"], n6=d["n6"],
            bonus=d["bonus"],
            prize_1=d.get("prize_1"),
            prize_2=d.get("prize_2"),
            prize_3=d.get("prize_3"),
            prize_4=d.get("prize_4"),
            prize_5=d.get("prize_5"),
            fetched_at=d.get("fetched_at"),
        )


@dataclass
class UserExtraction:
    session_id: str
    target_round: int
    ball_set: int
    strategy: int
    numbers: list
    id: Optional[int] = None
    user_name: Optional[str] = None      # 계정 이름 (미계정 시 None)
    phone_last4: Optional[str] = None    # 전화번호 뒷 4자리 (미계정 시 None)
    extracted_at: Optional[str] = None
    rank: Optional[int] = None
    match_count: Optional[int] = None
    bonus_match: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: dict) -> "UserExtraction":
        return cls(
            id=d.get("id"),
            session_id=d["session_id"],
            user_name=d.get("user_name"),
            phone_last4=d.get("phone_last4"),
            target_round=d["target_round"],
            ball_set=d["ball_set"],
            strategy=d["strategy"],
            numbers=d["numbers"],
            extracted_at=d.get("extracted_at"),
            rank=d.get("rank"),
            match_count=d.get("match_count"),
            bonus_match=d.get("bonus_match"),
        )


@dataclass
class WeeklyAnnouncement:
    round: int
    draw_date: date
    winning_numbers: dict
    stats: dict
    total_extractions: int = 0
    id: Optional[int] = None
    published_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "WeeklyAnnouncement":
        draw_date = d["draw_date"]
        if isinstance(draw_date, str):
            draw_date = date.fromisoformat(draw_date)
        return cls(
            id=d.get("id"),
            round=d["round"],
            draw_date=draw_date,
            winning_numbers=d["winning_numbers"],
            stats=d["stats"],
            total_extractions=d.get("total_extractions", 0),
            published_at=d.get("published_at"),
        )
