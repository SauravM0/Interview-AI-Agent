"""
Evaluation Agent — runs as a separate LiveKit agent worker.

Flow:
  1. Joins the interview room as a silent observer (no mic, no speaker)
  2. Listens on the "transcript" data channel — collects every message
  3. When it receives {"type": "interview_complete"} on "agent-control" channel,
     it calls the Groq evaluator and POSTs results to the API server
  4. Saves the evaluation to the DB via /api/interviews/complete

Run separately from the interview agent:
  python agent/evaluator_agent.py dev
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import cli

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation-agent")


async def evaluator_entrypoint(ctx: agents.JobContext) -> None:
    """
    Evaluation agent entrypoint.
    Silently observes the room, collects transcript, evaluates on completion.
    """
    logger.info("Evaluation agent joining room: %s", ctx.room.name)
    await ctx.connect()

    transcript_lines: list[dict] = []
    interview_id: Optional[str] = None
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

    # Fetch interview_id for this room from API
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{api_url}/api/agent/context/{ctx.room.name}"
            )
            if resp.status_code == 200:
                data = resp.json()
                interview_id = data.get("interview_id")
                logger.info(
                    "Evaluation agent connected for interview_id=%s", interview_id
                )
    except Exception as exc:
        logger.warning("Could not fetch interview context: %s", exc)

    # ── Data channel listener ─────────────────────────────────────────────────

    @ctx.room.on("data_received")
    def on_data(data_packet):
        try:
            topic = getattr(data_packet, "topic", None)
            payload = json.loads(bytes(data_packet.data).decode("utf-8"))
        except Exception:
            return

        if topic == "transcript":
            # Collect every transcript line the interview agent publishes
            transcript_lines.append({
                "speaker": payload.get("speaker", "Unknown"),
                "text": payload.get("text", ""),
                "timestamp": payload.get("timestamp", time.time()),
            })
            logger.debug(
                "Transcript line #%d: [%s] %s",
                len(transcript_lines),
                payload.get("speaker"),
                payload.get("text", "")[:60],
            )

        elif topic == "agent-control" and payload.get("type") == "interview_complete":
            logger.info(
                "Received interview_complete signal — starting evaluation "
                "(%d transcript lines collected)", len(transcript_lines)
            )
            asyncio.create_task(
                run_evaluation(transcript_lines, interview_id, api_url)
            )

    # Keep agent alive until room closes
    logger.info("Evaluation agent listening... (waiting for interview_complete signal)")
    await asyncio.sleep(7200)  # 2-hour max — room closes before this


async def run_evaluation(
    transcript_lines: list[dict],
    interview_id: Optional[str],
    api_url: str,
) -> None:
    """Run the Groq-powered evaluation and POST results to API server."""
    if not transcript_lines:
        logger.warning("No transcript lines collected — skipping evaluation")
        return

    # Format transcript as plain text for the evaluator LLM
    transcript_text = "\n".join(
        f"{line['speaker']}: {line['text']}"
        for line in transcript_lines
        if line.get("text", "").strip()
    )

    logger.info(
        "Evaluating transcript (%d chars, %d lines)...",
        len(transcript_text),
        len(transcript_lines),
    )

    from evaluation.evaluator import InterviewEvaluator
    evaluator = InterviewEvaluator()

    try:
        results = await evaluator.evaluate_interview(transcript_text)
        logger.info(
            "Evaluation complete — overall score: %s",
            results.get("overall_score"),
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        results = {
            "error": str(exc),
            "overall_score": 0,
            "report_markdown": f"# Evaluation Failed\n\n{exc}",
        }

    # Save results back to API server
    api_secret = os.getenv("API_SECRET_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_url}/api/interviews/complete",
                json={
                    "interview_id": interview_id,
                    "transcript": transcript_text,
                    "evaluation": results,
                },
                headers={"x-api-key": api_secret},
            )
            if resp.status_code == 200:
                logger.info(
                    "Evaluation saved for interview_id=%s", interview_id
                )
            else:
                logger.error(
                    "Failed to save evaluation (status=%s): %s",
                    resp.status_code,
                    resp.text[:300],
                )
    except Exception as exc:
        logger.error("Failed to POST evaluation to API: %s", exc)


if __name__ == "__main__":
    if os.getenv("ENABLE_LIVE_EVALUATOR") != "1":
        logger.error(
            "Live evaluator worker is disabled. Evaluation now runs after the "
            "interview via agent.main. Set ENABLE_LIVE_EVALUATOR=1 only for "
            "debugging the old silent observer mode."
        )
        raise SystemExit(1)

    missing = [
        v for v in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GROQ_API_KEY"]
        if not os.getenv(v)
    ]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        raise SystemExit(1)

    import socket as _socket

    def free_port(start: int = 8400) -> int:
        for p in range(start, start + 100):
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                    s.bind(("", p))
                    return p
            except OSError:
                continue
        return 0

    port = int(os.getenv("EVAL_AGENT_PORT", 0)) or free_port()
    logger.info("Starting evaluation agent on port %s", port)
    cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=evaluator_entrypoint, port=port)
    )
