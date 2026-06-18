"""
서버 자체 재학습 모듈 (m03)
===========================
절차:
  1. Supabase draw_results 에서 is_winning=true 만 fetch
  2. BayesianFrequencyModel.fit() (수 초)
  3. 원자적 파일 교체 (best_m03.npz.tmp → best_m03.npz)
  4. number_gen.reload_model() 호출

동시 실행 방지: threading.Lock
"""

import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytz

from .notify import notify_event as _notify

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model_m03_claude"))

from m03_model import BayesianFrequencyModel  # noqa: E402
from m03_train import fetch_history            # noqa: E402

logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")

CKPT_PATH = PROJECT_ROOT / "model_m03_claude" / "best_m03.npz"
META_PATH = PROJECT_ROOT / "model_m03_claude" / "training_meta.json"

_retrain_lock = threading.Lock()
_last_result: dict | None = None


def retrain(
    from_round: int | None = None,
    include_mock: bool = False,
    include_bonus: bool = False,
    alpha: float = 1.0,
    triggered_by: str = "scheduler",
) -> dict:
    """m03 재학습. 동시 실행 방지. 실패 시 기존 체크포인트 보존."""
    global _last_result

    if not _retrain_lock.acquire(blocking=False):
        return {"status": "skipped", "reason": "이미 재학습 진행 중"}

    start_ts = datetime.now(KST)
    result: dict = {
        "status": "running",
        "triggered_by": triggered_by,
        "started_at": start_ts.isoformat(),
        "config": {
            "from_round": from_round,
            "include_mock": include_mock,
            "include_bonus": include_bonus,
            "alpha": alpha,
        },
    }
    logger.info(f"[retrain] 시작 trigger={triggered_by} alpha={alpha}")

    try:
        # 1. 데이터 fetch
        df = fetch_history(from_round, include_mock)
        if len(df) == 0:
            raise RuntimeError("Supabase 에서 데이터가 없음")
        result["num_rows"] = len(df)
        result["round_range"] = [int(df["round"].min()), int(df["round"].max())]
        logger.info(
            f"[retrain] 데이터 {len(df)}행  회차 {df['round'].min()}~{df['round'].max()}"
        )

        # 2. 학습 (빈도 집계)
        model = BayesianFrequencyModel(alpha=alpha).fit(df, include_bonus)

        # 3. 원자적 파일 교체 (numpy.savez 는 .npz 확장자를 자동으로 붙임)
        tmp_stem = CKPT_PATH.parent / (CKPT_PATH.stem + ".tmp")
        model.save(str(tmp_stem))
        tmp_file = Path(str(tmp_stem) + ".npz")
        os.replace(tmp_file, CKPT_PATH)

        # 메타 저장
        import json
        meta = {
            "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "from_round": from_round,
            "include_mock": include_mock,
            "include_bonus": include_bonus,
            "alpha": alpha,
            "num_draws_global": model.num_draws_global,
            "num_draws_per_set": model.num_draws_per_set,
            "round_range": list(model.round_range),
            "triggered_by": triggered_by,
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 4. 메모리 리로드
        from .number_gen import reload_model
        info = reload_model()
        result["model_info"] = info

        end_ts = datetime.now(KST)
        result.update({
            "status": "success",
            "finished_at": end_ts.isoformat(),
            "duration_seconds": (end_ts - start_ts).total_seconds(),
        })
        logger.info(
            f"[retrain] 완료  {result['duration_seconds']:.2f}s  "
            f"회차 {model.round_range[0]}~{model.round_range[1]}"
        )
        _notify(
            "✅", "재학습 완료",
            f"회차 {model.round_range[0]}~{model.round_range[1]} · "
            f"{result['duration_seconds']:.1f}s · trigger={triggered_by}",
        )

    except Exception as e:
        logger.exception("[retrain] 실패")
        result.update({
            "status": "failed",
            "error": str(e),
            "finished_at": datetime.now(KST).isoformat(),
        })
        _notify("❌", "재학습 실패", f"{e} · trigger={triggered_by}")
    finally:
        _last_result = result
        _retrain_lock.release()

    return result


def get_last_result() -> dict | None:
    return _last_result
