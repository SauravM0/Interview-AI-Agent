"""
LiveKit client wrapper — token generation and room management.

FIXED:
  - timedelta import moved to top (was at bottom, causing NameError on every token call)
  - Lazy lkapi creation now logs URL clearly
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta          # FIX: was at the VERY BOTTOM of the file — NameError on every token call
from typing import Optional

from livekit import api

logger = logging.getLogger("LiveKitClient")

PLACEHOLDER_KEYS = {"replace_with_livekit_key", "replace_with_livekit_secret"}


class LiveKitClient:
    """Wrapper around livekit-api for managing rooms and tokens."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        livekit_url: Optional[str] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("LIVEKIT_API_KEY") or "").strip()
        self.api_secret = (api_secret or os.getenv("LIVEKIT_API_SECRET") or "").strip()
        self.livekit_url = (
            livekit_url or os.getenv("LIVEKIT_URL") or "http://localhost:7880"
        ).strip()
        self._lkapi: Optional[api.LiveKitAPI] = None

        if not self.api_key or not self.api_secret:
            logger.error(
                "LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set. LiveKit features will fail."
            )
            raise RuntimeError("LiveKit credentials are missing.")

        if any(ph in self.api_key for ph in PLACEHOLDER_KEYS) or any(
            ph in self.api_secret for ph in PLACEHOLDER_KEYS
        ):
            logger.error(
                "LiveKit credentials are still placeholders. Update LIVEKIT_API_KEY and LIVEKIT_API_SECRET in your .env."
            )
            raise RuntimeError(
                "LiveKit credentials are still placeholders (update .env)."
            )

        logger.info(
            "LiveKit client configured: url=%s api_key=%s",
            self.livekit_url,
            self.api_key,
        )

    @property
    def lkapi(self) -> Optional[api.LiveKitAPI]:
        """Lazy initialization of LiveKitAPI."""
        if self._lkapi is None:
            try:
                self._lkapi = api.LiveKitAPI(
                    self.livekit_url,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
            except Exception as exc:
                logger.error("Failed to initialize LiveKit API: %s", exc)
                return None
        return self._lkapi

    def generate_token(
        self,
        room_name: str,
        participant_identity: str,
        participant_name: str,
        role: str = "candidate",
        ttl_minutes: int = 60,
    ) -> str:
        """Generate a JWT access token for a participant."""
        if not self.api_key or not self.api_secret:
            raise ValueError("LiveKit credentials missing")

        grants = api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )

        if role == "observer_hr":
            grants.can_publish = False
            grants.can_publish_data = False

        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(participant_identity)
            .with_name(participant_name)
            .with_grants(grants)
            .with_ttl(timedelta(minutes=ttl_minutes))   # timedelta now imported at top
        )
        return token.to_jwt()

    async def create_room_async(
        self, room_name: str, empty_timeout: int = 600
    ) -> api.Room:
        if not self.lkapi:
            raise RuntimeError("LiveKit API not initialized")
        logger.info("Creating room: %s", room_name)
        return await self.lkapi.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=empty_timeout,
                max_participants=10,
            )
        )

    async def list_rooms(self):
        if not self.lkapi:
            return []
        return await self.lkapi.room.list_rooms(api.ListRoomsRequest())

    async def close_room(self, room_name: str) -> None:
        if not self.lkapi:
            return
        logger.info("Closing room: %s", room_name)
        await self.lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))

    async def validate_credentials(self) -> str:
        """Validate LiveKit credentials. Returns probe token if successful."""
        probe_token = self.generate_token(
            room_name="livekit-debug",
            participant_identity="debug-probe",
            participant_name="Debug Probe",
            role="observer_hr",
            ttl_minutes=5,
        )
        if not self.lkapi:
            raise RuntimeError(
                "LiveKit API client not initialized; check URL or credentials."
            )
        try:
            await self.lkapi.room.list_rooms(api.ListRoomsRequest())
        except Exception as exc:
            logger.error("LiveKit credential validation failed: %s", exc)
            raise
        return probe_token
