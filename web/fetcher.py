"""
동행복권 공식 API에서 추첨 결과 가져오기
API: https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={회차}
"""

import logging
from datetime import date

import requests
from sqlalchemy.orm import Session

from .database import DrawResult

logger = logging.getLogger(__name__)

LOTTO_API = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
TIMEOUT = 10


def fetch_draw(round_no: int) -> dict | None:
    """공식 API에서 특정 회차 추첨 결과 조회"""
    try:
        resp = requests.get(LOTTO_API.format(round_no), timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("returnValue") == "success":
            return {
                "round":    data["drwNo"],
                "draw_date": data["drwNoDate"],
                "numbers":  [data[f"drwtNo{i}"] for i in range(1, 7)],
                "bonus":    data["bnusNo"],
            }
    except Exception as e:
        logger.warning(f"[fetcher] {round_no}회차 조회 실패: {e}")
    return None


def get_latest_round() -> int:
    """
    날짜 기반으로 최신 회차 추정 후 실제 API로 확인
    로또 6/45 첫 회차: 2002-12-07
    """
    start = date(2002, 12, 7)
    estimated = (date.today() - start).days // 7 + 1

    for n in range(estimated + 3, estimated - 5, -1):
        result = fetch_draw(n)
        if result:
            return result["round"]

    return estimated


def save_draw_result(db: Session, data: dict) -> DrawResult:
    """추첨 결과 DB 저장 (이미 있으면 기존 반환)"""
    existing = db.query(DrawResult).filter(DrawResult.round == data["round"]).first()
    if existing:
        return existing

    nums = data["numbers"]
    draw = DrawResult(
        round=data["round"],
        draw_date=date.fromisoformat(data["draw_date"]),
        n1=nums[0], n2=nums[1], n3=nums[2],
        n4=nums[3], n5=nums[4], n6=nums[5],
        bonus=data["bonus"],
    )
    db.add(draw)
    db.commit()
    db.refresh(draw)
    logger.info(f"[fetcher] {data['round']}회차 저장 완료: {nums} + {data['bonus']}")
    return draw
