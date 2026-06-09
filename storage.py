"""
StorageService — saves transcripts and reports.
FIX: base_path is now relative to THIS file, not the process cwd.
     This was a bug when running from Docker / different directories.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Storage")

_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


class StorageService:
    def __init__(self, base_path: Optional[str] = None):
        root = Path(base_path) if base_path else _ROOT / "data" / "storage"
        self.transcripts_dir = root / "transcripts"
        self.reports_dir = root / "reports"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_transcript(self, interview_id: str, content: str) -> str:
        path = self.transcripts_dir / f"{interview_id}.json"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def save_report(self, interview_id: str, content_md: str) -> str:
        path = self.reports_dir / f"{interview_id}.md"
        path.write_text(content_md, encoding="utf-8")
        return str(path)

    def get_transcript(self, interview_id: str) -> Optional[str]:
        path = self.transcripts_dir / f"{interview_id}.json"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def get_report(self, interview_id: str) -> Optional[str]:
        path = self.reports_dir / f"{interview_id}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None
