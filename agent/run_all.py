"""
Run the LiveKit interview agent.

Usage:
  python agent/run_all.py start

This starts only the speaking interview worker. Post-interview evaluation is
triggered by the interview agent after the transcript is complete, so the
silent evaluator no longer competes for LiveKit room jobs.
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentOrchestrator")


def reexec_with_repo_venv() -> None:
    """Prefer the repo virtualenv when users launch with plain python."""
    if os.getenv("AGENT_SKIP_VENV_REEXEC") == "1":
        return
    project_root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        candidates = [project_root / ".venv" / "Scripts" / "python.exe"]
    else:
        candidates = [project_root / ".venv_wsl" / "bin" / "python"]
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current:
            logger.info("Re-running agent with project virtualenv: %s", candidate)
            os.environ["AGENT_SKIP_VENV_REEXEC"] = "1"
            os.execv(str(candidate), [str(candidate), *sys.argv])


def find_free_port(start: int) -> int:
    for port in range(start, start + 50):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found starting at {start}")


def check_env() -> None:
    required = [
        "GROQ_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        logger.error(
            "Missing required environment variables: %s\n"
            "Add them to your .env file and try again.",
            ", ".join(missing),
        )
        sys.exit(1)
    logger.info("Environment OK")


def main() -> None:
    reexec_with_repo_venv()

    from dotenv import load_dotenv

    load_dotenv()
    check_env()

    agent_dir = Path(__file__).parent
    interview_port = find_free_port(8300)
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    env = {**os.environ, "LIVEKIT_AGENT_PORT": str(interview_port)}

    logger.info("Starting interview agent in mode: %s", mode)
    interview_proc = subprocess.Popen(
        [sys.executable, str(agent_dir / "main.py"), mode],
        env=env,
    )
    logger.info(
        "Interview agent started (PID=%s port=%s)",
        interview_proc.pid,
        interview_port,
    )
    logger.info(
        "\n"
        "==========================================\n"
        "  Interview agent running.\n"
        "  Interview agent: port %s\n"
        "  Evaluation: runs after interview completion.\n"
        "  Press Ctrl+C to stop.\n"
        "==========================================",
        interview_port,
    )

    try:
        interview_proc.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down interview agent...")
    finally:
        interview_proc.terminate()
        interview_proc.wait(timeout=5)
        logger.info("Interview agent stopped.")


if __name__ == "__main__":
    main()
