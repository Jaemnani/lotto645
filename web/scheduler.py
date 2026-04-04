"""
APScheduler - 토요일 21:05 KST 자동 추첨 결과 수집 + 통계 계산
"""

import logging

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .database import SessionLocal
from .fetcher import fetch_draw, get_latest_round, save_draw_result
from .stats import calculate_and_save_stats

logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")


def saturday_job():
    """
    토요일 21:05 실행
    1. 동행복권 API에서 최신 회차 결과 가져오기
    2. DB 저장
    3. 사용자 번호 등수 계산 + 공지 생성
    """
    logger.info("[scheduler] 토요일 통계 작업 시작")
    db = SessionLocal()
    try:
        latest = get_latest_round()
        data   = fetch_draw(latest)

        if not data:
            logger.warning(f"[scheduler] {latest}회차 결과 아직 미발표 - 재시도 필요")
            return

        save_draw_result(db, data)
        ann = calculate_and_save_stats(db, latest)

        if ann:
            logger.info(f"[scheduler] {latest}회차 완료 | 통계: {ann.stats}")
        else:
            logger.info(f"[scheduler] {latest}회차 추첨 저장 완료 (추출자 없음)")

    except Exception:
        logger.exception("[scheduler] 오류 발생")
    finally:
        db.close()


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=KST)

    # 토요일 21:05 (추첨 발표 21:00 이후 5분 여유)
    scheduler.add_job(
        saturday_job,
        CronTrigger(day_of_week="sat", hour=21, minute=5, timezone=KST),
        id="saturday_stats",
        replace_existing=True,
        misfire_grace_time=3600,  # 1시간 내 놓친 실행 허용
    )

    logger.info("[scheduler] 토요일 21:05 KST 스케줄 등록 완료")
    return scheduler
