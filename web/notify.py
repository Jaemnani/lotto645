"""
Discord Webhook 알림 (서버 in-process 이벤트용)
==============================================
APScheduler 잡(saturday_job, hourly_retrain_check)과 retrain 은 쉘 래퍼가 없어
로그 파싱 방식(scripts/notify.sh)을 못 쓴다. 그래서 파이썬에서 직접 webhook 을 친다.

설계 (docs/05-ops-notifications.md):
  - 미설정 = 무동작. DISCORD_WEBHOOK_URL 없으면 조용히 no-op → 기존 동작 불변.
  - 실패 격리. 타임아웃 + 예외 삼킴 → 알림 장애가 본 작업(스케줄러 잡)을 절대 안 죽임.
  - 길이 truncate (Discord 2000자 제한).
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
# 메시지 앞에 붙일 출처 라벨 (Oracle / Synology 등 구분용). 미설정 시 hostname.
_SOURCE = os.environ.get("DEPLOY_ENV") or os.environ.get("HOSTNAME") or "server"


def discord_send(content: str) -> None:
    """DISCORD_WEBHOOK_URL 있으면 POST, 없으면 no-op. 절대 예외를 밖으로 던지지 않음."""
    if not _WEBHOOK:
        return
    try:
        body = f"`{_SOURCE}` · {content}"
        if len(body) > 1900:
            body = body[:1900] + "…(생략)"
        requests.post(_WEBHOOK, json={"content": body}, timeout=10)
    except Exception:
        # 알림 실패가 스케줄러 잡을 죽이면 안 됨 → 로그만 남기고 무시
        logger.warning("[notify] Discord 전송 실패 (무시)", exc_info=False)


def notify_event(emoji: str, title: str, detail: str = "") -> None:
    """간단 이벤트 헬퍼. 예: notify_event('✅', '재학습 완료', '회차 1100~1180 (3.2s)')"""
    msg = f"{emoji} **{title}**"
    if detail:
        msg += f" · {detail}"
    discord_send(msg)


def enabled() -> bool:
    return bool(_WEBHOOK)
