"""
동행복권 추첨 결과 가져오기

전략 (우선순위 순):
  1. smarPage HTML 파싱 → 최신 회차 번호/날짜/번호 한 번에 취득
  2. 공식 JSON API (특정 회차 조회용)
  3. 날짜 계산 fallback (회차 번호만)
"""

import logging
import re
from datetime import date

import requests
from bs4 import BeautifulSoup
from supabase import Client

from .database import DrawResult

logger = logging.getLogger(__name__)

SMAR_URL  = "https://www.dhlottery.co.kr/smarPage"
LOTTO_API = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
TIMEOUT   = 15
HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.dhlottery.co.kr/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def _fetch_smar() -> dict | None:
    """
    smarPage에서 최신 회차 전체 정보 파싱.

    HTML 구조:
      - 회차/날짜: 텍스트에서 정규식으로 추출
      - 번호:  div.result-ballBox 안의 div.result-ball
                figure 태그 이전 = 주번호 6개
                figure 태그 이후 = 보너스 1개
    """
    try:
        resp = requests.get(SMAR_URL, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # ── 회차 번호: div.round-num ───────────────────────────────
        # <div class="round-num">1218<span class="round-txt">회</span></div>
        round_tag = soup.find("div", class_="round-num")
        if not round_tag:
            logger.warning("[fetcher] round-num 태그 미발견")
            return None
        round_no = int(re.sub(r"\D", "", round_tag.get_text()))

        # ── 추첨일: div.today-date ─────────────────────────────────
        # <div class="today-date">2026년 04월 04일</div>
        date_tag = soup.find("div", class_="today-date")
        if not date_tag:
            logger.warning("[fetcher] today-date 태그 미발견")
            return None
        m = re.search(r"(\d{4})년\s*(\d{2})월\s*(\d{2})일", date_tag.get_text())
        draw_date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

        # ── 번호 파싱 ──────────────────────────────────────────────
        ball_box = soup.find("div", class_="result-ballBox")
        if not ball_box:
            logger.warning("[fetcher] result-ballBox 미발견")
            return None

        figure    = ball_box.find("figure")

        main_balls  = []
        bonus_balls = []
        passed_fig  = False

        for child in ball_box.children:
            if child == figure:
                passed_fig = True
                continue
            if hasattr(child, "get") and "result-ball" in child.get("class", []):
                n = int(child.get_text(strip=True))
                if passed_fig:
                    bonus_balls.append(n)
                else:
                    main_balls.append(n)

        if len(main_balls) != 6 or not bonus_balls:
            logger.warning(f"[fetcher] 번호 파싱 이상: main={main_balls}, bonus={bonus_balls}")
            return None

        logger.info(f"[fetcher] smarPage 파싱 성공: {round_no}회 {draw_date_str} {main_balls}+{bonus_balls[0]}")
        return {
            "round":     round_no,
            "draw_date": draw_date_str,
            "numbers":   main_balls,
            "bonus":     bonus_balls[0],
        }

    except Exception as e:
        logger.warning(f"[fetcher] smarPage 파싱 실패: {e}")
        return None


def get_latest_round() -> int:
    """최신 회차 번호 반환. smarPage 실패 시 날짜 계산 fallback."""
    data = _fetch_smar()
    if data and data["round"]:
        return data["round"]

    # fallback: 2002-12-07 첫 회차 기준 날짜 계산
    logger.warning("[fetcher] 날짜 계산으로 회차 추정")
    return (date.today() - date(2002, 12, 7)).days // 7 + 1


def fetch_draw(round_no: int) -> dict | None:
    """
    특정 회차 추첨 결과 조회.
    최신 회차면 smarPage 결과 재활용, 이전 회차면 JSON API 사용.
    """
    # 최신 회차는 smarPage에서 바로 가져옴
    latest = _fetch_smar()
    if latest and latest["round"] == round_no:
        return latest

    # 이전 회차는 JSON API
    try:
        resp = requests.get(LOTTO_API.format(round_no), timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("returnValue") == "success":
            return {
                "round":     data["drwNo"],
                "draw_date": data["drwNoDate"],
                "numbers":   [data[f"drwtNo{i}"] for i in range(1, 7)],
                "bonus":     data["bnusNo"],
            }
    except Exception as e:
        logger.warning(f"[fetcher] {round_no}회차 JSON API 실패: {e}")

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
    for field in ("prize_1", "prize_2", "prize_3", "prize_4", "prize_5"):
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
