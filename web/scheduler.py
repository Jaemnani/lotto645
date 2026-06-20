"""
APScheduler
  - 토요일 21:05 KST : 추첨 결과 수집 + 통계 계산 (saturday_job)
  - 매 시간 정각     : Supabase 신규 회차 폴링 → 있으면 m03 재학습 (hourly_retrain_check)
  - 기동 시 1회      : 모델 파일 없거나 최신 회차와 차이나면 즉시 재학습
"""

import logging

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta

from .database import get_supabase, get_supabase_admin
from .fetcher import fetch_draw, get_latest_round, save_draw_result
from .notify import notify_event
from .retrain import retrain
from .stats import calculate_and_save_stats

logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")


def _latest_round_in_db() -> int | None:
    sb = get_supabase()
    rows = (
        sb.table("draw_results")
        .select("round")
        .eq("is_winning", True)
        .not_.is_("ball_set", "null")   # ball_set 까지 채워진 것만 학습 대상
        .order("round", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return int(rows[0]["round"]) if rows else None


def hourly_retrain_check():
    """매 시간 정각: DB 최신 회차가 현재 모델보다 크면 재학습"""
    from .number_gen import get_model_info

    try:
        info = get_model_info()
        db_latest = _latest_round_in_db()
        if db_latest is None:
            logger.info("[scheduler] DB 에 학습 가능한 회차 없음")
            return

        if not info.get("loaded"):
            logger.info("[scheduler] 모델 미로드 상태 - 초기 학습 트리거")
            retrain(triggered_by="scheduler_initial")
            return

        model_latest = info["round_range"][1]
        if db_latest > model_latest:
            logger.info(
                f"[scheduler] 신규 회차 감지  {model_latest} → {db_latest}, 재학습 시작"
            )
            notify_event("🔁", "신규 회차 감지 → 재학습", f"{model_latest} → {db_latest}회")
            retrain(triggered_by="scheduler_hourly")
        else:
            logger.debug(f"[scheduler] 신규 데이터 없음 (모델={model_latest}, DB={db_latest})")
    except Exception as e:
        logger.exception("[scheduler] 시간당 체크 오류")
        notify_event("❌", "시간당 재학습 체크 오류", str(e))


def saturday_job():
    """
    토요일 21:05 실행
    1. 동행복권 API에서 최신 회차 결과 가져오기
    2. DB 저장
    3. 사용자 번호 등수 계산 + 공지 생성
    """
    logger.info("[scheduler] 토요일 통계 작업 시작")
    notify_event("▶️", "토요일 통계 작업 시작", "당첨번호 fetch + 등수 계산")
    try:
        db     = get_supabase_admin()   # draw_results/announcements 쓰기 → service 키 필요
        latest = get_latest_round()
        data   = fetch_draw(latest)

        if not data:
            logger.warning(f"[scheduler] {latest}회차 결과 아직 미발표 - 재시도 필요")
            notify_event("⚠️", "토요일 통계 보류", f"{latest}회차 결과 미발표 — 재시도 필요")
            return

        save_draw_result(db, data)
        ann = calculate_and_save_stats(db, latest)

        if ann:
            logger.info(f"[scheduler] {latest}회차 완료 | 통계: {ann.stats}")
            notify_event("✅", f"{latest}회차 통계 완료", str(ann.stats))
        else:
            logger.info(f"[scheduler] {latest}회차 추첨 저장 완료 (추출자 없음)")
            notify_event("✅", f"{latest}회차 추첨 저장 완료", "추출자 없음")

    except Exception as e:
        logger.exception("[scheduler] 오류 발생")
        notify_event("❌", "토요일 통계 작업 실패", str(e))


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=KST)

    # 토요일 21:05 (추첨 발표 21:00 이후 5분 여유)
    scheduler.add_job(
        saturday_job,
        CronTrigger(day_of_week="sat", hour=21, minute=5, timezone=KST),
        id="saturday_stats",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # 매 시간 정각 - 신규 회차 감지 시 재학습
    scheduler.add_job(
        hourly_retrain_check,
        CronTrigger(minute=0, timezone=KST),
        id="hourly_retrain",
        replace_existing=True,
        misfire_grace_time=1800,
        max_instances=1,
    )

    # 기동 후 30초 뒤 1회 (모델 없을 시 초기 학습 + 놓친 회차 캐치업)
    scheduler.add_job(
        hourly_retrain_check,
        DateTrigger(run_date=datetime.now(KST) + timedelta(seconds=30)),
        id="startup_retrain",
        replace_existing=True,
    )

    logger.info("[scheduler] saturday 21:05 + hourly retrain + startup 등록 완료")
    return scheduler
