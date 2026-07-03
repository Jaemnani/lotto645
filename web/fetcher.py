"""
동행복권 추첨 결과 가져오기

https://www.dhlottery.co.kr/lt645/result 회차 조회 API(selectPstLt645InfoNew.do)에서
번호 + 등수별 당첨금을 가져온다. 이 API는 result 페이지 방문으로 발급된 세션 쿠키 없이
호출하면 연결 자체가 차단된다 (WAF).

전략 (우선순위 순):
  1. selectPstLt645InfoNew.do (srchDir=center) → 회차 번호/날짜/번호/등수별 당첨금 취득
  2. 날짜 계산 fallback (get_latest_round 전용, 회차 번호만)
"""

import logging
import re
from datetime import date

import requests
from supabase import Client

from .database import DrawResult

logger = logging.getLogger(__name__)

RESULT_PAGE_URL = "https://www.dhlottery.co.kr/lt645/result"
ROUND_INFO_API  = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
# (connect, read) — 접속 자체가 막히는 경우 빨리 포기하고 재시도하도록 connect는 짧게
TIMEOUT   = (5, 15)
HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": RESULT_PAGE_URL,
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def open_session() -> requests.Session:
    """
    result 페이지 방문으로 세션 쿠키 확보. 브라우저에서 회차 이동 버튼을 눌렀을 때
    체감상 캐싱된 것처럼 빠른 이유는 이미 맺어둔 연결/세션을 그대로 재사용하기 때문 —
    매 조회마다 새 세션을 열면(연결을 새로 맺으면) 느려지고 차단 확률도 올라간다.
    여러 회차를 연달아 조회할 때는 이 세션 하나를 만들어 재사용할 것.
    """
    session = requests.Session()
    resp = session.get(RESULT_PAGE_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return session


def _parse_item(item: dict) -> dict:
    ymd = str(item["ltRflYmd"])
    return {
        "round":     item["ltEpsd"],
        "draw_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
        "numbers":   [item[f"tm{i}WnNo"] for i in range(1, 7)],
        "bonus":     item["bnsWnNo"],
        # 1게임당 당첨금
        "prize_1":   item.get("rnk1WnAmt"),
        "prize_2":   item.get("rnk2WnAmt"),
        "prize_3":   item.get("rnk3WnAmt"),
        "prize_4":   item.get("rnk4WnAmt"),
        "prize_5":   item.get("rnk5WnAmt"),
        # 등수별 당첨 게임 수
        "winners_1": item.get("rnk1WnNope"),
        "winners_2": item.get("rnk2WnNope"),
        "winners_3": item.get("rnk3WnNope"),
        "winners_4": item.get("rnk4WnNope"),
        "winners_5": item.get("rnk5WnNope"),
        # 등위별 총 당첨금
        "total_prize_1": item.get("rnk1SumWnAmt"),
        "total_prize_2": item.get("rnk2SumWnAmt"),
        "total_prize_3": item.get("rnk3SumWnAmt"),
        "total_prize_4": item.get("rnk4SumWnAmt"),
        "total_prize_5": item.get("rnk5SumWnAmt"),
        # 해당 회차 총 판매금액
        "total_sales": item.get("rlvtEpsdSumNtslAmt"),
    }


def get_latest_round(session: requests.Session | None = None) -> int:
    """최신 회차 번호 반환. result 페이지 회차선택 드롭박스 현재값(#opt_val) 파싱.
    실패 시 날짜 계산 fallback."""
    try:
        s = session or open_session()
        resp = s.get(RESULT_PAGE_URL, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        m = re.search(r'id="opt_val"\s+value="(\d+)"', resp.text)
        if m:
            return int(m.group(1))
        logger.warning("[fetcher] opt_val 태그 미발견")
    except Exception as e:
        logger.warning(f"[fetcher] 최신 회차 조회 실패: {e}")

    logger.warning("[fetcher] 날짜 계산으로 회차 추정")
    return (date.today() - date(2002, 12, 7)).days // 7 + 1


def fetch_draws_around(round_no: int, session: requests.Session | None = None) -> list[dict]:
    """
    srchLtEpsd=round_no 기준 center 조회 결과 전체(최대 10회차: 오래된5+선택1+최신4) 반환.
    한 번 호출로 여러 회차를 함께 얻을 수 있어 대량 백필 시 fetch_draw보다 이걸 직접 쓰면
    API 호출 수를 크게 줄일 수 있다.

    session을 넘기면 그 연결을 재사용한다 — 브라우저에서 회차 이동 버튼을 연달아 눌렀을 때
    체감상 캐싱된 것처럼 빠른 것과 같은 이유(연결 재사용)로, 여러 회차를 연속 조회할 때는
    매번 새 세션을 여는 것보다 훨씬 빠르고 차단될 확률도 낮다.
    """
    try:
        s = session or open_session()
        resp = s.get(
            ROUND_INFO_API,
            params={"srchDir": "center", "srchLtEpsd": round_no},
            headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("data", {}).get("list") or []
        return [_parse_item(item) for item in items]
    except Exception as e:
        logger.warning(f"[fetcher] {round_no}회차 주변 조회 실패: {e}")
        return []


def fetch_draw(round_no: int, session: requests.Session | None = None) -> dict | None:
    """특정 회차 추첨 결과 조회 (번호 + 등수별 당첨금)."""
    for data in fetch_draws_around(round_no, session=session):
        if data["round"] == round_no:
            logger.info(f"[fetcher] {round_no}회차 조회 성공: {data['numbers']}+{data['bonus']}")
            return data
    logger.warning(f"[fetcher] {round_no}회차가 응답 목록에 없음")
    return None


def save_draw_result(db: Client, data: dict) -> DrawResult:
    """
    공홈 추첨 결과 DB 저장 — 항상 is_winning=True (실제 당첨번호) 행만 다룸.

    - (round, is_winning=True) 행이 이미 있으면 → prize 등 추가 데이터 UPDATE 후 반환
    - 없으면 → INSERT (ball_set은 카페 크롤링 전까지 NULL)
    """
    nums = data["numbers"]
    draw_date = (
        date.fromisoformat(data["draw_date"])
        if data.get("draw_date")
        else date.today()
    )

    # 공홈에서 제공하는 추가 데이터 (있을 때만 포함)
    extra = {}
    for field in (
        "prize_1", "prize_2", "prize_3", "prize_4", "prize_5",
        "winners_1", "winners_2", "winners_3", "winners_4", "winners_5",
        "total_prize_1", "total_prize_2", "total_prize_3", "total_prize_4", "total_prize_5",
        "total_sales",
    ):
        if data.get(field) is not None:
            extra[field] = data[field]

    existing = (
        db.table("draw_results")
        .select("*")
        .eq("round", data["round"])
        .eq("is_winning", True)
        .execute()
        .data
    )

    if existing:
        # 카페에서 먼저 들어온 행 → prize 등 추가 데이터만 업데이트
        if extra:
            db.table("draw_results") \
              .update(extra) \
              .eq("round", data["round"]) \
              .eq("is_winning", True) \
              .execute()
            logger.info(f"[fetcher] {data['round']}회차 prize 업데이트: {extra}")
        result = (
            db.table("draw_results")
            .select("*")
            .eq("round", data["round"])
            .eq("is_winning", True)
            .execute()
            .data[0]
        )
        return DrawResult.from_dict(result)

    # 카페 크롤링 전 공홈이 먼저 → INSERT
    row = {
        "round":      data["round"],
        "draw_date":  draw_date.isoformat(),
        "is_winning": True,
        "n1": nums[0], "n2": nums[1], "n3": nums[2],
        "n4": nums[3], "n5": nums[4], "n6": nums[5],
        "bonus":      data["bonus"],
        **extra,
    }
    inserted = db.table("draw_results").insert(row).execute().data[0]
    logger.info(f"[fetcher] {data['round']}회차 저장 (is_winning=True): {nums} + {data['bonus']}")
    return DrawResult.from_dict(inserted)
