"""
등수 계산 및 주간 통계 생성
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from .database import DrawResult, UserExtraction, WeeklyAnnouncement

logger = logging.getLogger(__name__)

# 등수 레이블
RANK_LABELS = {1: "1등", 2: "2등", 3: "3등", 4: "4등", 5: "5등", 6: "낙첨"}


def calculate_rank(
    user_numbers: list[int],
    winning: list[int],
    bonus: int,
) -> tuple[int, int, bool]:
    """
    로또 6/45 등수 계산

    Returns
    -------
    (rank, match_count, bonus_match)
      rank: 1-5 당첨, 6 낙첨
    """
    winning_set = set(winning)
    user_set    = set(user_numbers)
    match       = len(user_set & winning_set)
    b_match     = bonus in user_set

    if   match == 6:              return 1, match, False
    elif match == 5 and b_match:  return 2, match, True
    elif match == 5:              return 3, match, False
    elif match == 4:              return 4, match, False
    elif match == 3:              return 5, match, False
    else:                         return 6, match, False


def calculate_and_save_stats(db: Session, round_no: int) -> WeeklyAnnouncement | None:
    """
    해당 회차 추첨 결과로 사용자 번호 등수 일괄 계산 후 공지 저장

    Returns
    -------
    WeeklyAnnouncement or None (추첨 결과 미존재 시)
    """
    draw = db.query(DrawResult).filter(DrawResult.round == round_no).first()
    if not draw:
        logger.warning(f"[stats] {round_no}회차 추첨 결과 없음")
        return None

    winning = [draw.n1, draw.n2, draw.n3, draw.n4, draw.n5, draw.n6]
    bonus   = draw.bonus

    # 아직 등수 미계산인 추출 번호만 처리
    extractions = (
        db.query(UserExtraction)
        .filter(
            UserExtraction.target_round == round_no,
            UserExtraction.rank.is_(None),
        )
        .all()
    )

    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for ext in extractions:
        rank, match, b_match   = calculate_rank(ext.numbers, winning, bonus)
        ext.rank               = rank
        ext.match_count        = match
        ext.bonus_match        = b_match
        rank_counts[rank]     += 1

    db.commit()

    stats = {
        RANK_LABELS[r]: rank_counts[r] for r in range(1, 7)
    }
    stats["total"] = len(extractions)

    # 이미 공지 있으면 업데이트
    existing = (
        db.query(WeeklyAnnouncement)
        .filter(WeeklyAnnouncement.round == round_no)
        .first()
    )
    if existing:
        existing.stats             = stats
        existing.total_extractions = len(extractions)
        existing.published_at      = datetime.utcnow()
        db.commit()
        logger.info(f"[stats] {round_no}회차 공지 업데이트: {stats}")
        return existing

    ann = WeeklyAnnouncement(
        round             = round_no,
        draw_date         = draw.draw_date,
        winning_numbers   = {"numbers": winning, "bonus": bonus},
        stats             = stats,
        total_extractions = len(extractions),
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    logger.info(f"[stats] {round_no}회차 공지 생성: {stats}")
    return ann
