"""
Supabase PostgreSQL 연결 및 SQLAlchemy 모델 정의
환경변수 DATABASE_URL 에 Supabase connection string 설정 필요
"""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Integer, JSON, String, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # 연결 끊김 자동 감지
    pool_recycle=300,          # 5분마다 연결 재사용
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DrawResult(Base):
    """동행복권 공식 추첨 결과"""
    __tablename__ = "draw_results"

    id         = Column(Integer, primary_key=True)
    round      = Column(Integer, unique=True, index=True, nullable=False)
    draw_date  = Column(Date, nullable=False)
    n1         = Column(Integer, nullable=False)
    n2         = Column(Integer, nullable=False)
    n3         = Column(Integer, nullable=False)
    n4         = Column(Integer, nullable=False)
    n5         = Column(Integer, nullable=False)
    n6         = Column(Integer, nullable=False)
    bonus      = Column(Integer, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class UserExtraction(Base):
    """사용자가 추출한 번호"""
    __tablename__ = "user_extractions"

    id           = Column(Integer, primary_key=True)
    session_id   = Column(String, index=True, nullable=False)
    target_round = Column(Integer, index=True, nullable=False)  # 응모 대상 회차
    extracted_at = Column(DateTime, default=datetime.utcnow)
    ball_set     = Column(Integer, nullable=False)  # 1-5
    strategy     = Column(Integer, nullable=False)  # 1-4
    numbers      = Column(JSON, nullable=False)     # [n1, n2, n3, n4, n5, n6]
    rank         = Column(Integer, nullable=True)   # 추첨 후 계산 (1-5, 6=낙첨)
    match_count  = Column(Integer, nullable=True)
    bonus_match  = Column(Boolean, nullable=True)


class WeeklyAnnouncement(Base):
    """주간 통계 공지사항"""
    __tablename__ = "weekly_announcements"

    id                = Column(Integer, primary_key=True)
    round             = Column(Integer, unique=True, index=True, nullable=False)
    draw_date         = Column(Date, nullable=False)
    winning_numbers   = Column(JSON, nullable=False)  # {"numbers": [...], "bonus": n}
    stats             = Column(JSON, nullable=False)   # {"1등": n, ..., "낙첨": n, "total": n}
    total_extractions = Column(Integer, default=0)
    published_at      = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
