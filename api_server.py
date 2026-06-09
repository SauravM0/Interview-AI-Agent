"""
FastAPI orchestrator server.

CRITICAL FIXES applied (all bugs that break the running product):

1. STARTUP CRASH — timedelta NameError in livekit_client.py
   Root cause: `from datetime import timedelta` was at the BOTTOM of livekit_client.py,
   after generate_token() that uses it. Python reads imports lazily at call-time only
   inside functions, but the class body references it at class-definition time via
   .with_ttl(timedelta(...)). This caused an immediate NameError the first time a token
   was ever generated — which happens at startup validation. Fixed in livekit_client.py.

2. INTERVIEW SETTINGS LOST ON AGENT START — in-memory store only
   Root cause: interview_settings_store was a plain dict in the API process. The agent
   process is a SEPARATE process. It fetches settings via HTTP — which works IF the
   API server is in the same process, but the store resets on any restart/redeploy.
   Fix: persist settings into the interviews.interview_settings DB column on POST/PUT.

3. list_candidates BROKEN — generator used as Session
   Root cause: `db: Session = db_session()` — db_session() is a generator function.
   Calling it returns a generator object, not a Session. The endpoint then called
   .query() on a generator, raising AttributeError at runtime.
   Fix: use `with SessionLocal() as db:` (already done for other endpoints).

4. CORS blocks agent fetches in production — only localhost:3000 allowed
   Root cause: allow_origins was ['http://localhost:3000', 'http://127.0.0.1:3000'].
   The Python agent uses httpx, not a browser, so CORS doesn't apply there, but the
   frontend served from any non-localhost origin (Vercel, custom domain) will be
   blocked. Fix: read origins from ALLOWED_ORIGINS env var with a sane default.

5. SECURITY — debug endpoints are public
   Root cause: /api/debug/db and /api/debug/livekit expose credentials, DB path,
   process ID. Fix: gate them behind verify_api_key dependency.

6. SECURITY — API_SECRET_TOKEN leaks into DB URL log
   Root cause: debug_db() replaced API_SECRET_TOKEN in engine.url string — but only
   if token was non-empty. It did str replacement on the full URL which could match
   partial substrings. Replaced with a static mask.

7. RESUME UPLOAD — duplicate db_session generator call pattern
   Root cause: The retry loop created a new SessionLocal() on each iteration without
   closing the previous one on continue (the finally only ran on break). Refactored
   to always close in finally.

8. Missing import: `timedelta` was never imported in api_server.py but was used in
   /api/debug/livekit-auth. Fixed by adding the import.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import socket
import sqlite3
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta          # FIX #8: timedelta was missing
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from pypdf import PdfReader
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIServer")

load_dotenv()

DEV_MODE = os.getenv("ENV", "").lower() == "development"


def require_env(var_name: str, placeholders: Optional[List[str]] = None) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error("Missing required environment variable: %s", var_name)
        raise RuntimeError(
            f"{var_name} is required but not set. Please configure it in your .env file."
        )
    normalized = value.strip()
    if placeholders:
        for placeholder in placeholders:
            if placeholder in normalized:
                logger.error(
                    "%s is using placeholder value (%s); please set a real credential.",
                    var_name,
                    placeholder,
                )
                raise RuntimeError(
                    f"{var_name} must be set to your real LiveKit credential (update .env)."
                )
    return normalized


LIVEKIT_URL = require_env("LIVEKIT_URL")
LIVEKIT_API_KEY = require_env("LIVEKIT_API_KEY", ["replace_with_livekit_key"])
LIVEKIT_API_SECRET = require_env(
    "LIVEKIT_API_SECRET", ["replace_with_livekit_secret"]
)
API_SECRET_TOKEN = require_env("API_SECRET_TOKEN")

# FIX #4: configurable CORS origins
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


def to_ws_url(url: str) -> str:
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    return url


LIVEKIT_WS_URL = to_ws_url(LIVEKIT_URL)
logger.info(
    "LiveKit startup config: url=%s api_key=%s (ws_url=%s)",
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_WS_URL,
)


def to_json_text(obj: Any) -> Optional[str]:
    """Convert Python object to JSON string for DB storage."""
    if obj is None:
        return None
    return json.dumps(obj)


def from_json_text(text: Optional[str]) -> Any:
    """Parse JSON string from DB to Python object."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

from livekit_client import LiveKitClient
from scheduler.workflow import InterviewScheduler
from scheduler.database import (
    Candidate,
    Interview,
    InterviewStatus,
    SessionLocal,
    engine,
    init_db,
)
from storage import StorageService

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Interview AI Orchestrator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # FIX #4
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ── Services ──────────────────────────────────────────────────────────────────

scheduler = InterviewScheduler()
lk_client = LiveKitClient(
    api_key=LIVEKIT_API_KEY,
    api_secret=LIVEKIT_API_SECRET,
    livekit_url=LIVEKIT_URL,
)


# ── LiveKit startup probe ─────────────────────────────────────────────────────

def validate_livekit_on_startup() -> None:
    def tcp_check(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def to_http_url(url: str) -> str:
        if url.startswith("ws://"):
            return "http://" + url[len("ws://"):]
        if url.startswith("wss://"):
            return "https://" + url[len("wss://"):]
        return url

    http_base = to_http_url(LIVEKIT_URL).rstrip("/")
    validate_url = http_base + "/rtc/validate"
    logger.info(
        "Validating LiveKit credentials at %s (api_key=%s)",
        validate_url,
        LIVEKIT_API_KEY,
    )

    try:
        token = lk_client.generate_token(
            room_name="sanity-check",
            participant_identity="startup-probe",
            participant_name="Startup Probe",
            role="observer_hr",
            ttl_minutes=2,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LiveKit credential check failed while generating token: {exc}"
        ) from exc

    params = {
        "access_token": token,
        "auto_subscribe": 1,
        "sdk": "js",
        "version": "2.16.1",
        "protocol": 16,
    }
    deadline = time.monotonic() + 30.0
    delays = [0.5, 1.0, 2.0]
    delay_index = 0
    last_error: Optional[Exception] = None

    while True:
        try:
            resp = httpx.get(validate_url, params=params, timeout=5.0)
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_error = exc
            is_tcp_up = tcp_check("127.0.0.1", 7880)
            logger.warning(
                "LiveKit validation error: %s: %s (tcp_7880=%s)",
                exc.__class__.__name__,
                exc,
                "open" if is_tcp_up else "closed",
            )
            if DEV_MODE:
                logger.warning("Hint: docker logs recruitment-ai-livekit-1 --tail 200")
        else:
            if resp.status_code == 200:
                logger.info("LiveKit credentials validated successfully.")
                return
            if resp.status_code in {401, 403}:
                raise RuntimeError(
                    "LiveKit rejected token. Check livekit.yaml keys and .env match."
                )
            last_error = RuntimeError(
                f"LiveKit returned status {resp.status_code} during validation."
            )
            logger.warning(
                "LiveKit validation non-200 (status=%s body=%s)",
                resp.status_code,
                resp.text[:300],
            )

        if time.monotonic() >= deadline:
            break

        delay = delays[min(delay_index, len(delays) - 1)]
        delay_index += 1
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))

    raise RuntimeError(
        "LiveKit not reachable or crashing; check docker logs recruitment-ai-livekit-1"
    ) from last_error


validate_livekit_on_startup()

# ── Pydantic models ───────────────────────────────────────────────────────────

class ScheduleRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    resume_text: Optional[str] = None


class TokenRequest(BaseModel):
    room_name: str
    identity: str
    name: str
    role: str = "candidate"


class CloseRoomRequest(BaseModel):
    room_name: str


class InterviewSettingsPayload(BaseModel):
    target_role: str
    experience_level: str
    interview_type: str
    focus_areas: List[str] = []
    duration_minutes: Optional[int] = None
    duration: Optional[str] = None
    preferred_language: str
    room_name: Optional[str] = None
    session_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_session_key(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("room_name") and not values.get("session_id"):
                raise ValueError("room_name or session_id is required")
        return values

    @model_validator(mode="after")
    def normalize_duration(self) -> "InterviewSettingsPayload":
        if self.duration_minutes is None and self.duration:
            match = re.search(r"\d+", self.duration)
            if match:
                return self.model_copy(
                    update={"duration_minutes": int(match.group(0))}
                )
        return self


# In-memory fallback cache (agent reads from DB first; this is a fast path)
interview_settings_store: Dict[str, Dict[str, Any]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_resume_text(
    text: str, candidate_name: str, candidate_email: str
) -> Dict[str, Any]:
    text = text.replace("\u2022", "-")

    def normalize_section(header: str) -> str:
        header = header.lower().strip()
        if "experience" in header:
            return "experience"
        if "project" in header:
            return "projects"
        if "education" in header:
            return "education"
        if "skill" in header:
            return "skills"
        if "summary" in header or "profile" in header:
            return "summary"
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: Dict[str, List[str]] = {}
    current = "summary"
    for line in lines:
        if re.fullmatch(r"[A-Za-z &/]{3,}", line):
            section = normalize_section(line)
            if section:
                current = section
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(line)

    contact: Dict[str, str] = {}
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)
    if email_match:
        contact["email"] = email_match.group(0)
    if phone_match:
        contact["phone"] = phone_match.group(0)

    summary_text = " ".join(sections.get("summary", [])[:3]).strip()
    skills: List[str] = []
    if "skills" in sections:
        raw = " ".join(sections["skills"])
        skills = [
            item.strip()
            for item in re.split(r"[,|/]+", raw)
            if item.strip()
        ]

    def parse_bullets(source: List[str]) -> List[str]:
        return [entry.lstrip("-* ").strip() for entry in source if entry.strip()]

    experience: List[Dict[str, Any]] = []
    if "experience" in sections:
        block: List[str] = []
        for line in sections["experience"]:
            if line and line.isupper() and block:
                block = []
            block.append(line)
            if len(block) >= 6:
                _push_experience(block, experience, parse_bullets)
                block = []
        if block:
            _push_experience(block, experience, parse_bullets)

    projects: List[Dict[str, Any]] = []
    if "projects" in sections:
        block = []
        for line in sections["projects"]:
            if line.isupper() and block:
                block = []
            block.append(line)
            if len(block) >= 5:
                projects.append(
                    {
                        "name": block[0],
                        "tech": "",
                        "bullets": parse_bullets(block[1:]),
                        "outcomes": "",
                    }
                )
                block = []
        if block:
            projects.append(
                {
                    "name": block[0],
                    "tech": "",
                    "bullets": parse_bullets(block[1:]),
                    "outcomes": "",
                }
            )

    education = sections.get("education", [])[:6]
    return {
        "name": candidate_name,
        "contact": contact,
        "summary": summary_text,
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "education": education,
        "raw_sections": list(sections.keys()),
    }


def _push_experience(block, experience, parse_bullets):
    company_role = block[0]
    role = company = ""
    if " - " in company_role:
        role, company = [p.strip() for p in company_role.split(" - ", 1)]
    elif " | " in company_role:
        company, role = [p.strip() for p in company_role.split(" | ", 1)]
    else:
        company = company_role
    experience.append(
        {
            "company": company,
            "role": role,
            "dates": block[1] if len(block) > 1 else "",
            "bullets": parse_bullets(block[2:]),
        }
    )


def interview_to_response(interview: Interview) -> Dict[str, Any]:
    data = {
        col.name: getattr(interview, col.name)
        for col in Interview.__table__.columns
    }
    data["interview_settings"] = from_json_text(data.get("interview_settings"))
    data["resume_json"] = from_json_text(data.get("resume_json"))
    return data


# ── Security ──────────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/api/debug/db", dependencies=[Depends(verify_api_key)])   # FIX #5: gated
def debug_db():
    db_path = engine.url.database
    db_exists = bool(db_path and os.path.exists(db_path))
    db_size = os.path.getsize(db_path) if db_exists else -1

    tables: List[str] = []
    interview_info: List[str] = []
    journal_mode = synchronous = busy_timeout = "unknown"

    try:
        import sqlalchemy

        with engine.connect() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    sqlalchemy.text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                )
            ]
            try:
                interview_info = [
                    str(r)
                    for r in conn.execute(
                        sqlalchemy.text("PRAGMA table_info(interviews)")
                    )
                ]
            except Exception:
                interview_info = ["error getting table info"]
            journal_mode = conn.execute(
                sqlalchemy.text("PRAGMA journal_mode")
            ).scalar()
            synchronous = conn.execute(
                sqlalchemy.text("PRAGMA synchronous")
            ).scalar()
            busy_timeout = conn.execute(
                sqlalchemy.text("PRAGMA busy_timeout")
            ).scalar()
    except Exception as exc:
        logger.error("Debug DB error: %s", exc)

    return {
        "db_url": "sqlite:///[redacted]",          # FIX #6: no credential leak
        "db_path": db_path,
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "process_id": os.getpid(),
        "sqlite_version": sqlite3.sqlite_version,
        "tables": tables,
        "pragma_table_info_interviews": interview_info,
        "pragma_journal_mode": journal_mode,
        "pragma_synchronous": synchronous,
        "pragma_busy_timeout": busy_timeout,
    }


@app.get("/api/debug/livekit", dependencies=[Depends(verify_api_key)])   # FIX #5
async def debug_livekit():
    try:
        token = await lk_client.validate_credentials()
        return {"ok": True, "token": token, "url": lk_client.livekit_url}
    except Exception as exc:
        logger.error("LiveKit credentials check failed: %s", exc)
        return Response(
            content=json.dumps({"ok": False, "error": str(exc)}),
            status_code=500,
            media_type="application/json",
        )


@app.get("/api/debug/livekit-auth", dependencies=[Depends(verify_api_key)])  # FIX #5
def debug_livekit_auth():
    room_name = "livekit-auth-check"
    ttl_minutes = 5
    try:
        token = lk_client.generate_token(
            room_name=room_name,
            participant_identity="auth-checker",
            participant_name="Auth Checker",
            role="observer_hr",
            ttl_minutes=ttl_minutes,
        )
        expires_at = (
            datetime.utcnow() + timedelta(minutes=ttl_minutes)
        ).isoformat() + "Z"
        return {
            "ok": True,
            "iss": LIVEKIT_API_KEY,
            "room_name": room_name,
            "expires_at": expires_at,
        }
    except Exception as exc:
        logger.error("LiveKit auth token generation failed: %s", exc)
        return Response(
            content=json.dumps(
                {"ok": False, "error": f"LiveKit token generation failed: {exc}"}
            ),
            status_code=500,
            media_type="application/json",
        )


@app.post("/api/interviews/schedule", dependencies=[Depends(verify_api_key)])
def schedule_interview(req: ScheduleRequest):
    try:
        interview = scheduler.schedule_interview(
            req.candidate_name, req.candidate_email
        )
        return {
            "success": True,
            "interview_id": interview.id,
            "room_name": interview.room_name,
            "meet_link": interview.meet_link,
            "scheduled_time": interview.scheduled_time,
        }
    except Exception as exc:
        logger.error("Schedule error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "resume"

    # Parse content
    text_content = ""
    if filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            text_content = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            logger.error("PDF parse error: %s", exc)
            text_content = "Could not parse PDF content."
    else:
        text_content = content.decode("utf-8", errors="ignore")

    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text_content)
    candidate_email = (
        email_match.group(0)
        if email_match
        else f"candidate_{uuid.uuid4().hex[:8]}@example.com"
    )
    candidate_name = (
        filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    )

    backoffs = [0.2, 0.6, 1.2]
    for attempt in range(3):
        db = SessionLocal()
        try:
            existing = (
                db.query(Candidate).filter(Candidate.email == candidate_email).first()
            )
            if existing:
                existing.resume_text = text_content
                existing.resume_path = filename
                existing.name = candidate_name
                candidate = existing
            else:
                candidate = Candidate(
                    name=candidate_name,
                    email=candidate_email,
                    resume_text=text_content,
                    resume_path=filename,
                )
                db.add(candidate)

            db.flush()

            room_name = f"interview-{uuid.uuid4().hex[:8]}"
            resume_json = parse_resume_text(
                text_content, candidate_name, candidate_email
            )
            interview = Interview(
                candidate_id=candidate.id,
                room_name=room_name,
                status=InterviewStatus.SCHEDULED,
                scheduled_time=datetime.utcnow(),
                meet_link=f"https://meet.livekit.io/{room_name}",
                interview_settings=json.dumps({}),
                resume_json=json.dumps(resume_json),
            )
            db.add(interview)
            db.commit()

            token = lk_client.generate_token(
                room_name=room_name,
                participant_identity=candidate.email,
                participant_name=candidate.name,
                role="candidate",
            )
            return {
                "success": True,
                "room_name": room_name,
                "token": token,
                "url": LIVEKIT_WS_URL,
                "identity": candidate.email,
                "candidate_name": candidate.name,
            }

        except IntegrityError as exc:
            db.rollback()
            logger.exception("Integrity error during resume upload: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=f"Database constraint failed: {getattr(exc, 'orig', exc)}",
            )
        except (OperationalError, sqlite3.OperationalError) as exc:
            db.rollback()
            msg = str(exc).lower()
            if "database is locked" in msg and attempt < 2:
                logger.warning(
                    "Database locked (attempt %s/3); retrying: %s", attempt + 1, exc
                )
                time.sleep(backoffs[attempt])
                continue
            if "no such column" in msg or "no column named" in msg:
                raise HTTPException(
                    status_code=500,
                    detail="DB schema mismatch: run migration or reset db",
                )
            raise HTTPException(
                status_code=500,
                detail=f"Database write failed: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            db.rollback()
            logger.exception("Unexpected error during resume upload: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Upload error: {type(exc).__name__}: {exc}",
            )
        finally:
            db.close()   # FIX #7: always close, even on continue

    raise HTTPException(status_code=500, detail="Database locked after 3 retries")


@app.get("/api/agent/context/{room_name}")
def get_agent_context(room_name: str):
    with SessionLocal() as db:
        interview = (
            db.query(Interview).filter(Interview.room_name == room_name).first()
        )
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        candidate = (
            db.query(Candidate)
            .filter(Candidate.id == interview.candidate_id)
            .first()
        )
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Merge DB-persisted settings with in-memory cache (DB wins)
        db_settings = from_json_text(interview.interview_settings) or {}
        cached_settings = interview_settings_store.get(room_name, {})
        merged_settings = {**cached_settings, **db_settings}

        return {
            "interview_id": interview.id,
            "candidate_name": candidate.name,
            "job_title": "General Software Engineer",
            "interview_settings": merged_settings,
            "resume_profile": from_json_text(interview.resume_json) or {},
        }


def _persist_settings(session_key: str, payload: InterviewSettingsPayload) -> None:
    """Write settings into DB interview row AND in-memory cache."""
    settings_dict = payload.dict()
    interview_settings_store[session_key] = settings_dict   # fast path for agent

    # FIX #2: persist to DB so agent gets them even after API restart
    with SessionLocal() as db:
        interview = (
            db.query(Interview).filter(Interview.room_name == session_key).first()
        )
        if interview:
            interview.interview_settings = json.dumps(settings_dict)
            db.commit()
        else:
            logger.warning(
                "Settings saved to memory only — no interview row found for room %s",
                session_key,
            )


@app.post("/api/interviews/settings")
def set_interview_settings(req: InterviewSettingsPayload):
    session_key = req.room_name or req.session_id
    if not session_key:
        raise HTTPException(status_code=400, detail="room_name or session_id is required")
    _persist_settings(session_key, req)
    return {"ok": True}


@app.put("/api/interviews/settings")
def update_interview_settings(req: InterviewSettingsPayload):
    return set_interview_settings(req)


@app.options("/api/interviews/settings")
def options_interview_settings():
    return Response(status_code=200)


@app.get("/api/interviews/settings")
def get_interview_settings(
    room_name: Optional[str] = None, session_id: Optional[str] = None
):
    session_key = room_name or session_id
    if not session_key:
        raise HTTPException(
            status_code=400, detail="room_name or session_id is required"
        )

    # Try DB first, fall back to in-memory cache
    with SessionLocal() as db:
        interview = (
            db.query(Interview).filter(Interview.room_name == session_key).first()
        )
        if interview:
            db_settings = from_json_text(interview.interview_settings)
            if db_settings:
                return db_settings

    cached = interview_settings_store.get(session_key)
    if cached:
        return cached

    raise HTTPException(status_code=404, detail="settings not found")


@app.post("/api/token")
def generate_token(req: TokenRequest):
    try:
        token = lk_client.generate_token(
            room_name=req.room_name,
            participant_identity=req.identity,
            participant_name=req.name,
            role=req.role,
        )
        return {"token": token, "url": LIVEKIT_WS_URL}
    except Exception as exc:
        logger.exception(
            "Failed to generate token for room %s identity %s", req.room_name, req.identity
        )
        raise HTTPException(status_code=500, detail="Failed to generate LiveKit token")


@app.get("/api/interviews/generate-token")
def generate_token_get(
    room_name: str, identity: str, name: str, role: str = "candidate"
):
    try:
        token = lk_client.generate_token(
            room_name=room_name,
            participant_identity=identity,
            participant_name=name,
            role=role,
        )
        return {"token": token, "url": LIVEKIT_WS_URL, "room_name": room_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/interviews/complete", dependencies=[Depends(verify_api_key)])
async def complete_interview(req: Dict[str, Any]):
    interview_id = req.get("interview_id")
    transcript_text = req.get("transcript", "")
    evaluation_data = req.get("evaluation")

    if not interview_id:
        raise HTTPException(status_code=400, detail="interview_id is required")

    with SessionLocal() as db:
        interview = db.query(Interview).filter(Interview.id == interview_id).first()
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")

        interview.status = InterviewStatus.COMPLETED
        db.commit()

    # If the evaluation agent already ran and sent results, save them
    if evaluation_data:
        from scheduler.database import Evaluation
        overall_score = evaluation_data.get("overall_score", 0)
        scores = evaluation_data.get("scores", {})
        report_md = evaluation_data.get("report_markdown", "")

        storage = StorageService()
        report_path = storage.save_report(interview_id, report_md)

        with SessionLocal() as db:
            existing = db.query(Evaluation).filter(
                Evaluation.interview_id == interview_id
            ).first()
            if existing:
                existing.overall_score = overall_score
                existing.score_json = json.dumps(scores)
                existing.summary = report_md[:500]
            else:
                eval_record = Evaluation(
                    interview_id=interview_id,
                    overall_score=overall_score,
                    score_json=json.dumps(scores),
                    summary=report_md[:500],
                )
                db.add(eval_record)

            interview_rec = db.query(Interview).filter(
                Interview.id == interview_id
            ).first()
            if interview_rec:
                interview_rec.status = InterviewStatus.EVALUATED

            db.commit()

        return {
            "status": "evaluated",
            "interview_id": interview_id,
            "overall_score": overall_score,
            "report_path": report_path,
        }

    # If no evaluation yet, just mark completed and return
    return {"status": "completed", "interview_id": interview_id}


@app.get("/api/interviews/{interview_id}/report")
def get_report(interview_id: str):
    storage = StorageService()
    content = storage.get_report(interview_id)
    if not content:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"content": content}


@app.get("/api/candidates")
def list_candidates(skip: int = 0, limit: int = 100):
    # FIX #3: original called db_session() as if it returned a Session (it's a generator)
    with SessionLocal() as db:
        candidates = db.query(Candidate).offset(skip).limit(limit).all()
        return [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "created_at": c.created_at,
            }
            for c in candidates
        ]


@app.get("/api/interviews")
def list_interviews(skip: int = 0, limit: int = 100):
    with SessionLocal() as db:
        interviews = db.query(Interview).offset(skip).limit(limit).all()
        return [interview_to_response(i) for i in interviews]


@app.get("/api/interviews/{interview_id}")
def get_interview(interview_id: str):
    with SessionLocal() as db:
        interview = (
            db.query(Interview).filter(Interview.id == interview_id).first()
        )
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        return interview_to_response(interview)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT") or os.getenv("API_PORT", 8001))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
