"""
Database — supports both SQLite (local dev) and PostgreSQL (Supabase / Render).
Switch via DATABASE_URL env var:
  SQLite (default):   sqlite:///./data/interviews.db
  Supabase Postgres:  postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
"""
from __future__ import annotations

import enum
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, String, Text, create_engine, event, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()
logger = logging.getLogger(__name__)


def _json_dump(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _json_load(v: Any) -> Any:
    if v is None or isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


class InterviewStatus(enum.Enum):
    CREATED = "CREATED"
    SCHEDULED = "SCHEDULED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    EVALUATED = "EVALUATED"
    CANCELLED = "CANCELLED"


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    resume_path = Column(String)
    resume_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    interviews = relationship("Interview", back_populates="candidate")


class Interview(Base):
    __tablename__ = "interviews"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String, ForeignKey("candidates.id"))
    candidate_name = Column(String)
    candidate_email = Column(String)
    room_name = Column(String, unique=True)
    status = Column(Enum(InterviewStatus), default=InterviewStatus.CREATED)
    scheduled_time = Column(DateTime)
    meet_link = Column(String)
    interview_settings = Column(Text)
    resume_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    candidate = relationship("Candidate", back_populates="interviews")
    evaluation = relationship("Evaluation", back_populates="interview", uselist=False)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    interview_id = Column(String, ForeignKey("interviews.id"))
    overall_score = Column(Float)
    score_json = Column(Text)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    interview = relationship("Interview", back_populates="evaluation")


@event.listens_for(Interview, "before_insert")
@event.listens_for(Interview, "before_update")
def _serialize_interview(mapper, connection, target: Interview):
    target.interview_settings = _json_dump(target.interview_settings)
    target.resume_json = _json_dump(target.resume_json)


# ── Engine setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    _data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(_data_dir, exist_ok=True)
    DATABASE_URL = f"sqlite:///{os.path.join(_data_dir, 'interviews.db')}"
    logger.info("No DATABASE_URL set — using SQLite at %s", DATABASE_URL)
else:
    # Supabase / Render Postgres: fix the scheme if needed
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("Using PostgreSQL database")

_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=_connect_args)
SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


@event.listens_for(engine, "connect")
def _set_pragmas(dbapi_conn, _record):
    if _is_sqlite:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.close()


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    logger.info("Initialising database schema...")
    Base.metadata.create_all(engine)
    if _is_sqlite:
        _migrate_sqlite()
    logger.info("Database ready.")


_REQUIRED_COLS = {
    "interview_settings": "TEXT", "resume_json": "TEXT",
    "candidate_name": "TEXT", "candidate_email": "TEXT",
    "meet_link": "TEXT", "updated_at": "TEXT",
}

def _migrate_sqlite() -> None:
    with engine.begin() as conn:
        existing = {r[1] for r in conn.execute(text("PRAGMA table_info(interviews)"))}
        for col, col_type in _REQUIRED_COLS.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE interviews ADD COLUMN {col} {col_type}"))
                logger.info("Added column: interviews.%s", col)
