"""
FastAPI 메인 앱
- 번호 생성 / 저장 API
- 추첨 결과 / 공지사항 조회 API
- 관리자용 수동 트리거 API
"""

import logging
import os
import uuid
from typing import Optional

import pytz
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import get_supabase
from .fetcher import fetch_draw, get_latest_round, save_draw_result
from .number_gen import generate_numbers
from .scheduler import create_scheduler
from .stats import calculate_and_save_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")

# ── 앱 초기화 ──────────────────────────────────────────────────────────────────
app = FastAPI(title="로또 번호 추출기", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# 스케줄러 시작
_scheduler = create_scheduler()
_scheduler.start()

# 정적 파일 (빌드된 React 앱)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────
def _require_admin(x_admin_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("ADMIN_KEY", "")
    if expected and x_admin_key != expected:
        raise HTTPException(403, "관리자 키 불일치")


# ── SPA 서빙 ───────────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str = ""):
    index = os.path.join(STATIC_DIR, "index.html")
    if full_path.startswith("api/") or full_path.startswith("admin/"):
        raise HTTPException(404)
    if os.path.isfile(index):
        return FileResponse(index)
    return JSONResponse({"message": "API 서버 정상 동작 중"})


# ── 스키마 ─────────────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    ball_set: int = 0   # 0=자동, 1-5
    strategy: int = 1   # 1-4
    save: bool = False


class ExtractResponse(BaseModel):
    numbers:      list[int]
    ball_set:     int
    strategy:     int
    target_round: int
    saved:        bool


# ── API: 번호 생성 ─────────────────────────────────────────────────────────────
@app.post("/api/extract", response_model=ExtractResponse)
def extract(
    req: ExtractRequest,
    response: Response,
    session_id: Optional[str] = Cookie(default=None),
):
    """번호 생성 (save=true 시 DB 저장)"""
    if not 0 <= req.ball_set <= 5:
        raise HTTPException(400, "ball_set은 0-5 사이여야 합니다")
    if not 1 <= req.strategy <= 4:
        raise HTTPException(400, "strategy는 1-4 사이여야 합니다")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id, max_age=365 * 24 * 3600, httponly=True)

    result       = generate_numbers(ball_set=req.ball_set, strategy=req.strategy)
    latest       = get_latest_round()
    target_round = latest + 1

    saved = False
    if req.save:
        db = get_supabase()
        db.table("user_extractions").insert({
            "session_id":   session_id,
            "target_round": target_round,
            "ball_set":     result["ball_set"],
            "strategy":     result["strategy"],
            "numbers":      result["numbers"],
        }).execute()
        saved = True
        logger.info(f"[api] 번호 저장: {result['numbers']} ({target_round}회차)")

    return ExtractResponse(
        numbers      = result["numbers"],
        ball_set     = result["ball_set"],
        strategy     = result["strategy"],
        target_round = target_round,
        saved        = saved,
    )


# ── API: 추첨 결과 ─────────────────────────────────────────────────────────────
@app.get("/api/draw/latest")
def latest_draw():
    """가장 최근 저장된 추첨 결과 (실제 당첨번호)"""
    db   = get_supabase()
    rows = (
        db.table("draw_results")
        .select("*")
        .eq("is_winning", True)
        .order("round", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        return {"round": None, "message": "추첨 결과 없음"}
    draw = rows[0]
    return {
        "round":     draw["round"],
        "draw_date": draw["draw_date"],
        "numbers":   [draw["n1"], draw["n2"], draw["n3"], draw["n4"], draw["n5"], draw["n6"]],
        "bonus":     draw["bonus"],
        "ball_set":  draw.get("ball_set"),
    }


# ── API: 공지사항 ──────────────────────────────────────────────────────────────
@app.get("/api/announcement/latest")
def latest_announcement():
    """최신 주간 통계 공지"""
    db   = get_supabase()
    rows = (
        db.table("weekly_announcements")
        .select("*")
        .order("round", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        return {"round": None, "message": "아직 공지사항이 없습니다"}
    return rows[0]


@app.get("/api/announcement/{round_no}")
def announcement_by_round(round_no: int):
    """특정 회차 공지"""
    db   = get_supabase()
    rows = (
        db.table("weekly_announcements")
        .select("*")
        .eq("round", round_no)
        .execute()
        .data
    )
    if not rows:
        raise HTTPException(404, f"{round_no}회차 공지사항 없음")
    return rows[0]


@app.get("/api/announcements")
def announcements_list(limit: int = 10):
    """최근 공지 목록"""
    db   = get_supabase()
    rows = (
        db.table("weekly_announcements")
        .select("*")
        .order("round", desc=True)
        .limit(min(limit, 50))
        .execute()
        .data
    )
    return rows


# ── API: 관리자 ────────────────────────────────────────────────────────────────
@app.post("/api/admin/fetch-and-calc")
def admin_fetch_and_calc(
    round_no: Optional[int] = None,
    _: None = Depends(_require_admin),
):
    """
    추첨 결과 수집 + 통계 계산 (관리자 전용)
    round_no 미지정 시 최신 회차 자동 조회
    """
    db     = get_supabase()
    target = round_no if round_no else get_latest_round()
    data   = fetch_draw(target)
    if not data:
        raise HTTPException(404, f"{target}회차 결과 조회 실패 (미발표 또는 네트워크 오류)")

    draw = save_draw_result(db, data)
    ann  = calculate_and_save_stats(db, target)

    return {
        "round":   target,
        "numbers": [draw.n1, draw.n2, draw.n3, draw.n4, draw.n5, draw.n6],
        "bonus":   draw.bonus,
        "stats":   ann.stats if ann else None,
    }
