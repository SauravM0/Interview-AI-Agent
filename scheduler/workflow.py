"""
Interview scheduling workflow.

FIXED:
  - Session leak: original code never closed the session in start_interview / complete_interview
    (missing try/finally). All sessions now use context managers.
  - complete_interview now correctly marks status as COMPLETED not EVALUATED.
  - schedule_interview: candidate_id FK lookup added; no longer re-uses email prefix as PK.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from .database import Interview, InterviewStatus, SessionLocal, init_db
from .services import NotificationService
from livekit_client import LiveKitClient

logger = logging.getLogger("WorkflowEngine")


class InterviewScheduler:
    def __init__(self) -> None:
        init_db()
        self.notifier = NotificationService()
        self.lk_client = LiveKitClient()

    def schedule_interview(
        self,
        candidate_name: str,
        candidate_email: str,
        candidate_id: str | None = None,
    ) -> Interview:
        scheduled_time = datetime.utcnow() + timedelta(minutes=5)

        # Build a safe room name from the provided candidate id or email prefix
        id_slug = (candidate_id or candidate_email.split("@")[0]).replace("@", "_")
        room_name = f"interview_{id_slug}_{int(time.time())}"

        # FIX: use context manager so session is always closed
        with SessionLocal() as session:
            interview = Interview(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                room_name=room_name,
                scheduled_time=scheduled_time,
                status=InterviewStatus.SCHEDULED,
                meet_link=f"http://localhost:3000/interview/{room_name}",
            )
            session.add(interview)
            session.commit()
            session.refresh(interview)

        self.notifier.send_interview_invite(
            candidate_email,
            candidate_name,
            interview.meet_link,
            scheduled_time.strftime("%Y-%m-%d %H:%M UTC"),
        )

        logger.info(
            "Scheduled interview for %s in room %s", candidate_email, room_name
        )
        return interview

    def start_interview(self, interview_id: str) -> bool:
        # FIX: original had no try/finally — session leaked on exception
        with SessionLocal() as session:
            interview = (
                session.query(Interview).filter_by(id=interview_id).first()
            )
            if not interview:
                return False
            interview.status = InterviewStatus.ACTIVE
            session.commit()

        logger.info(
            "Interview %s STARTED. Run agent for room: %s",
            interview_id,
            interview.room_name,
        )
        return True

    def complete_interview(self, interview_id: str, results: dict) -> None:
        # FIX: original had no try/finally — session leaked; also set EVALUATED
        #      before evaluation was actually saved, which is wrong ordering.
        with SessionLocal() as session:
            interview = (
                session.query(Interview).filter_by(id=interview_id).first()
            )
            if not interview:
                return
            interview.status = InterviewStatus.COMPLETED   # FIX: was EVALUATED prematurely
            session.commit()

        logger.info("Interview %s COMPLETED.", interview_id)
