"""
Recruitment AI Agent — LiveKit AgentSession

KEY CHANGES from original:
  1. TTS: Deepgram via official LiveKit plugin
  2. LLM: Groq llama-3.3-70b-versatile (free, 14,400 calls/day)
  3. STT: Groq Whisper (free)
  4. data_received handler fixed: single DataPacket arg (was 4 args — broke End Turn)
  5. user_input_transcribed event name fixed (was "user_speech_transcription")
  6. publish_data uses keyword args (was positional — broke data channel)
  7. Runs post-interview evaluation after the transcript is complete
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# FIX: Python 3.14 compatibility - set environment before any imports
# This ensures spawned subprocesses will have event loops
if sys.version_info >= (3, 11):
    import multiprocessing
    if os.name == 'nt':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, cli, llm as llm_module
from livekit.agents._exceptions import APIStatusError
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

# FIX: import Groq plugin
try:
    from livekit.plugins import groq as groq_plugin
except ImportError:
    groq_plugin = None
    logging.warning("livekit-plugins-groq not installed.")

# Optional Kokoro TTS support. There is no official PyPI LiveKit Kokoro plugin,
# so Deepgram is the default working TTS provider below.
try:
    from livekit.plugins import kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

try:
    from livekit.plugins import deepgram as deepgram_plugin
    DEEPGRAM_AVAILABLE = True
except ImportError:
    deepgram_plugin = None
    DEEPGRAM_AVAILABLE = False
    logging.warning("livekit-plugins-deepgram not installed — Deepgram TTS will be unavailable.")

# FIX: import Silero VAD (optional)
try:
    from livekit.plugins import silero
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    silero = None
    logging.warning("livekit-plugins-silero not installed — using default VAD.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recruitment-agent")
load_dotenv()

FALLBACK_MESSAGE = (
    "I'm having trouble generating a response right now. "
    "Please give me a moment and continue."
)


# ── Resilient LLM with circuit breaker ───────────────────────────────────────

class UnavailableLLM(llm_module.LLM):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def chat(
        self,
        *,
        chat_ctx: llm_module.ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions | None = None,
        **kwargs,
    ) -> llm_module.LLMStream:
        return _UnavailableLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
            message=self._message,
        )


class _UnavailableLLMStream(llm_module.LLMStream):
    def __init__(
        self,
        llm: llm_module.LLM,
        *,
        chat_ctx: llm_module.ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        message: str,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._message = message

    async def _run(self):
        await self._event_ch.send(llm_module.ChatChunk(
            id="", delta=llm_module.ChoiceDelta(role="assistant", content=self._message)
        ))


class ResilientLLM(llm_module.LLM):
    def __init__(self, inner: llm_module.LLM, fallback_message: str, status_callback=None):
        super().__init__()
        self._inner = inner
        self._fallback_message = fallback_message
        self._failure_times: deque[float] = deque()
        self._breaker_open_until = 0.0
        self._status_callback = status_callback

    def _breaker_open(self) -> bool:
        return time.monotonic() < self._breaker_open_until

    def _record_failure(self) -> None:
        now = time.monotonic()
        self._failure_times.append(now)
        while self._failure_times and now - self._failure_times[0] > 120:
            self._failure_times.popleft()
        if len(self._failure_times) >= 3:
            self._breaker_open_until = now + 120
            self._failure_times.clear()
            logger.warning("Circuit breaker opened for 120s")
            if self._status_callback:
                self._status_callback("error", "LLM temporarily unavailable")

    def chat(
        self,
        *,
        chat_ctx: llm_module.ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ) -> llm_module.LLMStream:
        if self._breaker_open():
            return _UnavailableLLMStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=conn_options,
                message=self._fallback_message,
            )
        return _ResilientStream(
            self._inner, self,
            chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options,
            fallback_message=self._fallback_message,
            record_failure=self._record_failure,
            status_cb=self._status_callback,
            chat_kwargs=kwargs,
        )


class _ResilientStream(llm_module.LLMStream):
    def __init__(
        self,
        inner: llm_module.LLM,
        llm: llm_module.LLM,
        *,
        chat_ctx: llm_module.ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        fallback_message: str,
        record_failure,
        status_cb=None,
        chat_kwargs: dict | None = None,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._inner = inner
        self._tools = tools
        self._chat_kwargs = chat_kwargs or {}
        self._fallback_message = fallback_message
        self._record_failure = record_failure
        self._status_cb = status_cb

    async def _run(self):
        backoffs = [0.5, 1.5]
        for attempt in range(3):
            try:
                stream = self._inner.chat(
                    chat_ctx=self._chat_ctx,
                    tools=self._tools,
                    conn_options=self._conn_options,
                    **self._chat_kwargs,
                )
                async for chunk in stream:
                    await self._event_ch.send(chunk)
                return
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code in (429, 500, 503) or status_code is None
                logger.warning("LLM attempt %s failed (status=%s): %s", attempt + 1, status_code, exc)
                if attempt == 2 or not retryable:
                    self._record_failure()
                    await self._event_ch.send(llm_module.ChatChunk(
                        id="",
                        delta=llm_module.ChoiceDelta(
                            role="assistant",
                            content=self._fallback_message,
                        ),
                    ))
                    return
                await asyncio.sleep(backoffs[attempt])


class SerialLLM(llm_module.LLM):
    """Wraps an LLM to serialize all chat() calls with a reentrancy-aware lock.

    Prevents concurrent LLM invocations from:
      - VoicePipelineAgent's internal flow (after on_speech())
      - handle_end_turn() -> respond_to()
    """

    def __init__(self, inner: llm_module.LLM, lock: asyncio.Lock):
        super().__init__()
        self._inner = inner
        self._lock = lock
        self._owner_id: int | None = None

    def chat(
        self,
        *,
        chat_ctx: llm_module.ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ) -> llm_module.LLMStream:
        return _SerialStream(
            self._inner,
            self,
            lock=self._lock,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            chat_kwargs=kwargs,
        )


class _SerialStream(llm_module.LLMStream):
    def __init__(
        self,
        inner: llm_module.LLM,
        llm: llm_module.LLM,
        *,
        lock: asyncio.Lock,
        chat_ctx: llm_module.ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        chat_kwargs: dict | None = None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._inner = inner
        self._lock = lock
        self._tools = tools
        self._chat_kwargs = chat_kwargs or {}

    async def _run(self):
        async with self._lock:
            stream = self._inner.chat(
                chat_ctx=self._chat_ctx,
                tools=self._tools,
                conn_options=self._conn_options,
                **self._chat_kwargs,
            )
            async for chunk in stream:
                await self._event_ch.send(chunk)


async def llm_self_test(llm: llm_module.LLM, model_name: str) -> bool:
    ctx = llm_module.ChatContext().append(role="user", text="Reply with the single word OK")
    try:
        stream = llm.chat(chat_ctx=ctx)
        async for _ in stream:
            break
        logger.info("LLM self-test passed for model %s", model_name)
        return True
    except Exception as exc:
        logger.error("LLM self-test FAILED for model %s: %s", model_name, exc)
        return False


# ── Interview state ───────────────────────────────────────────────────────────

@dataclass
class InterviewState:
    start_time: float
    current_phase: str = "A"
    question_index: int = 0
    asked_questions: list = field(default_factory=list)
    candidate_answers: list = field(default_factory=list)
    follow_up_pending: bool = False
    follow_up_reason: str = ""
    last_tech_mention: str = ""
    phase_targets: dict = field(default_factory=dict)

    def add_question(self, text: str) -> None:
        norm = " ".join(text.lower().strip().split())[:200]
        if norm and norm not in self.asked_questions:
            self.asked_questions.append(norm)

    def add_answer(self, text: str) -> None:
        if text.strip():
            self.candidate_answers.append(text.strip())

    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.start_time) / 60.0

    def recent_answers_summary(self) -> str:
        recent = self.candidate_answers[-3:]
        return " | ".join(recent) if recent else "No answers yet."


def is_vague_answer(answer: str) -> bool:
    if len(answer.split()) < 18:
        return True
    vague = ["not sure", "maybe", "kind of", "some", "various", "etc", "stuff", "things"]
    return any(m in answer.lower() for m in vague)


def extract_tech_mention(answer: str, focus_areas: list) -> str:
    keywords = [
        "python", "javascript", "react", "node", "django", "flask", "fastapi",
        "aws", "gcp", "azure", "docker", "kubernetes", "sql", "postgres", "mysql",
        "redis", "graphql", "typescript", "system design", "microservices",
        "machine learning", "ml", "ai", "llm",
    ] + [a.lower() for a in focus_areas]
    low = answer.lower()
    for kw in keywords:
        if kw in low:
            return kw
    return ""


# ── Agent entrypoint ──────────────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext) -> None:
    logger.info("Agent starting for room: %s", ctx.room.name)
    await ctx.connect()

    import httpx
    candidate_name = "Candidate"
    interview_id: Optional[str] = None
    interview_settings: dict = {}
    resume_profile: dict = {}
    settings_loaded = False

    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            ctx_resp = await client.get(f"{api_url}/api/agent/context/{ctx.room.name}")
            if ctx_resp.status_code == 200:
                data = ctx_resp.json()
                interview_id = data.get("interview_id")
                candidate_name = data.get("candidate_name", "Candidate")
                interview_settings = data.get("interview_settings") or {}
                resume_profile = data.get("resume_profile") or {}

            settings_resp = await client.get(
                f"{api_url}/api/interviews/settings",
                params={"room_name": ctx.room.name},
            )
            if settings_resp.status_code == 200:
                interview_settings = settings_resp.json() or {}
                settings_loaded = True
    except Exception as exc:
        logger.error("Failed to fetch agent context: %s", exc)

    logger.info("Waiting for participant to join...")
    participant = await ctx.wait_for_participant()
    logger.info("Participant joined: %s", participant.identity)

    # ── Data channel helpers ──────────────────────────────────────────────────

    async def publish_data(payload: dict, topic: str) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic=topic,
            )
        except Exception as exc:
            logger.error("publish_data failed (%s): %s", topic, exc)

    def publish_status(state: str, detail: Optional[str] = None) -> None:
        p: dict = {"type": "status", "state": state, "timestamp": time.time()}
        if detail:
            p["detail"] = detail
        asyncio.create_task(publish_data(p, "agent-status"))

    transcript_lines: list[dict] = []
    evaluation_started = False

    async def run_post_interview_evaluation() -> None:
        nonlocal evaluation_started
        if evaluation_started:
            return
        evaluation_started = True
        if not interview_id:
            logger.warning("Cannot run evaluation: interview_id is missing")
            return
        if not transcript_lines:
            logger.warning("Cannot run evaluation: transcript is empty")
            return
        publish_status("evaluating")
        try:
            from agent.evaluator_agent import run_evaluation

            await run_evaluation(transcript_lines, interview_id, api_url)
            publish_status("completed")
        except Exception as exc:
            logger.error("Post-interview evaluation failed: %s", exc)
            publish_status("error", f"Evaluation failed: {exc}")

    def publish_transcript(speaker: str, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        transcript_lines.append({
            "speaker": speaker,
            "text": clean_text,
            "timestamp": time.time(),
        })
        asyncio.create_task(publish_data(
            {
                "type": "transcript",
                "speaker": speaker,
                "text": clean_text,
                "timestamp": transcript_lines[-1]["timestamp"],
            },
            "transcript",
        ))

    async def signal_interview_complete() -> None:
        """
        Publish interview_complete and start post-interview evaluation.
        """
        await publish_data(
            {"type": "interview_complete", "room": ctx.room.name, "timestamp": time.time()},
            "agent-control",
        )
        logger.info("Published interview_complete signal for room %s", ctx.room.name)
        asyncio.create_task(run_post_interview_evaluation())

    # ── Build STT / TTS / LLM / VAD ──────────────────────────────────────────

    stt = groq_plugin.STT(model="whisper-large-v3-turbo")

    tts_provider = os.getenv("TTS_PROVIDER", "deepgram").lower()
    if tts_provider == "kokoro" and KOKORO_AVAILABLE:
        tts = kokoro.TTS()
        logger.info("Using Kokoro TTS")
    else:
        if not DEEPGRAM_AVAILABLE:
            logger.error("Deepgram TTS not available — install livekit-plugins-deepgram")
            raise RuntimeError("livekit-plugins-deepgram is required but not installed")
        if not os.getenv("DEEPGRAM_API_KEY"):
            logger.error("DEEPGRAM_API_KEY is required for Deepgram TTS")
            raise RuntimeError("DEEPGRAM_API_KEY is required for Deepgram TTS")
        tts_model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en")
        tts = deepgram_plugin.TTS(model=tts_model)
        logger.info("Using Deepgram TTS model: %s", tts_model)

    logger.info("Using LLM: llama-3.3-70b-versatile (Groq)")

    class GroqLLM(groq_plugin.LLM):
        def __init__(self, model: str):
            super().__init__(model=model)
            self._pending_state: str | None = None

        def set_pending_state(self, text: str) -> None:
            self._pending_state = text

        def chat(
            self,
            *,
            chat_ctx: llm_module.ChatContext,
            tools: list | None = None,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
            **kwargs,
        ):
            # AgentSession manages chat history in LiveKit Agents 1.x. Per-turn
            # guidance is passed through generate_reply(instructions=...).
            self._pending_state = None
            return super().chat(
                chat_ctx=chat_ctx, tools=tools, conn_options=conn_options, **kwargs
            )

    base_llm = GroqLLM(model="llama-3.3-70b-versatile")
    llm_ok = True  # Skip self-test for now
    if not llm_ok:
        base_llm = UnavailableLLM(FALLBACK_MESSAGE)
        logger.warning("LLM unavailable — using fallback responder for this session")

    llm_lock = asyncio.Lock()
    llm = SerialLLM(
        ResilientLLM(base_llm, FALLBACK_MESSAGE, status_callback=publish_status),
        lock=llm_lock,
    )
    
    # Load VAD
    if SILERO_AVAILABLE:
        vad = silero.VAD.load()
    else:
        vad = None  # Use default VAD from LiveKit

    # ── Interview configuration ───────────────────────────────────────────────

    target_role = interview_settings.get("target_role", "Software Engineer")
    experience_level = interview_settings.get("experience_level", "mid-level")
    interview_type = interview_settings.get("interview_type", "Mixed")
    focus_areas: list = interview_settings.get("focus_areas", [])
    duration_minutes: int = int(interview_settings.get("duration_minutes", 20))
    preferred_language: str = interview_settings.get("preferred_language", "English")
    focus_text = ", ".join(focus_areas) if focus_areas else "general engineering"
    top_skills = ", ".join((resume_profile.get("skills") or [])[:6]) or "Not provided"

    exp_highlights: list[str] = []
    for item in (resume_profile.get("experience") or [])[:2]:
        title = " — ".join(p for p in [item.get("role"), item.get("company")] if p)
        if title:
            exp_highlights.append(title)
    highlights_text = "\n    - ".join(exp_highlights) if exp_highlights else "No detailed experience provided."

    no_settings_note = "" if settings_loaded else (
        "\nNOTE: No interview settings were loaded. Start by asking the candidate "
        "what role they are targeting and what areas to focus on.\n"
    )

    system_prompt = f"""You are Eve, a professional and friendly AI recruiter conducting an initial screening interview.
Speak only in {preferred_language}. If the candidate switches languages, gently redirect.

INTERVIEW CONFIGURATION:
- Target role: {target_role}
- Experience level: {experience_level}
- Interview type: {interview_type}
- Focus areas: {focus_text}
- Duration: {duration_minutes} minutes
{no_settings_note}
CANDIDATE:
- Name: {candidate_name}
- Top skills: {top_skills}
- Experience:
    - {highlights_text}

FLOW:
Phase A — Warm welcome, confirm role, explain format (1 question)
Phase B — Resume deep-dive: 3 targeted questions about their specific experience
Phase C — Skills round: 3–5 questions based on focus areas
Phase D — Behavioral: 2 STAR-format questions
Phase E — Candidate questions + wrap-up

RULES:
1. Ask ONE question at a time. Wait for the full answer before continuing.
2. Ask follow-up questions when answers are vague or mention interesting technology.
3. Reference their specific experience and projects from the resume.
4. Be warm, encouraging, and professional.
5. Keep track of time — if nearing {duration_minutes} minutes, move to wrap-up.
"""

    state = InterviewState(
        start_time=time.monotonic(),
        phase_targets={"A": 1, "B": 3, "C": min(5, max(3, len(focus_areas) + 2)), "D": 2, "E": 1},
    )
    last_user_text = ""
    last_commit_at = 0.0
    partial_text = ""
    end_turn_at = 0.0
    end_turn_lock = asyncio.Lock()

    interviewer = Agent(instructions=system_prompt)
    session = AgentSession(
        vad=vad, stt=stt, llm=llm, tts=tts,
        min_endpointing_delay=0.4,
        allow_interruptions=True,
    )
    await session.start(interviewer, room=ctx.room)
    publish_status("connected")

    # ── Phase transitions ─────────────────────────────────────────────────────

    def advance_phase() -> None:
        elapsed = state.elapsed_minutes()
        if elapsed >= duration_minutes:
            state.current_phase = "E"
            state.question_index = 0
            return
        if elapsed >= duration_minutes * 0.8 and state.current_phase not in ("D", "E"):
            state.current_phase = "D"
            state.question_index = 0
            return
        if elapsed >= duration_minutes * 0.55 and state.current_phase not in ("C", "D", "E"):
            state.current_phase = "C"
            state.question_index = 0
            return
        transitions = {"A": "B", "B": "C", "C": "D", "D": "E"}
        phase = state.current_phase
        target = state.phase_targets.get(phase, 1)
        if state.question_index >= target and phase in transitions:
            state.current_phase = transitions[phase]
            state.question_index = 0

    def push_state(next_action: str) -> None:
        asked = "\n".join(f"  - {q}" for q in state.asked_questions[-6:]) or "  (none yet)"
        state_text = (
            f"CURRENT STATE:\n"
            f"  Phase: {state.current_phase}  |  Q index: {state.question_index}  "
            f"|  Elapsed: {state.elapsed_minutes():.1f}/{duration_minutes} min\n"
            f"  Follow-up pending: {state.follow_up_pending} ({state.follow_up_reason})\n"
            f"  Recent candidate answers: {state.recent_answers_summary()}\n"
            f"  Already asked (avoid repeating):\n{asked}\n"
            f"  NEXT ACTION: {next_action}"
        )
        base_llm.set_pending_state(state_text)

    # ── Data channel: End Turn button ─────────────────────────────────────────

    @ctx.room.on("data_received")
    def on_data_received(data_packet):
        # FIX: livekit-agents >=1.x passes a single DataPacket object, not 4 args
        if getattr(data_packet, "topic", None) != "agent-control":
            return
        try:
            msg = json.loads(bytes(data_packet.data).decode("utf-8"))
        except Exception:
            return
        if msg.get("type") == "end_turn":
            asyncio.create_task(handle_end_turn())

    async def handle_end_turn() -> None:
        nonlocal last_commit_at, end_turn_at, partial_text, last_user_text
        async with end_turn_lock:
            now = time.monotonic()
            if now - last_commit_at < 0.8:
                return
            end_turn_at = now
            forced = (partial_text or last_user_text or state.recent_answers_summary()).strip()
            if not forced:
                forced = "Please continue."
            last_user_text = forced
            last_commit_at = now
            publish_transcript(candidate_name, forced)
            state.add_answer(forced)
            state.follow_up_pending = False
            advance_phase()
            push_state("Respond to the candidate's input and ask the next appropriate question.")
            publish_status("thinking")
            handle = session.generate_reply(
                user_input=forced,
                instructions="Respond to the candidate's input and ask the next appropriate question.",
                allow_interruptions=True,
            )
            await handle.wait_for_playout()

    # ── Agent events ──────────────────────────────────────────────────────────

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        nonlocal last_user_text, last_commit_at, partial_text
        text = (getattr(event, "transcript", "") or "").strip()
        if not text:
            return
        partial_text = text
        if not getattr(event, "is_final", False):
            return
        if end_turn_at and time.monotonic() - end_turn_at < 2.0 and text == last_user_text:
            return
        last_user_text = text
        last_commit_at = time.monotonic()
        partial_text = ""
        publish_transcript(candidate_name, text)
        publish_status("thinking")
        state.add_answer(text)

        state.follow_up_pending = is_vague_answer(text)
        state.follow_up_reason = "Vague answer - ask for specifics." if state.follow_up_pending else ""
        tech = extract_tech_mention(text, focus_areas)
        if tech:
            state.follow_up_pending = True
            state.follow_up_reason = f"Candidate mentioned {tech} - ask one depth question."
            state.last_tech_mention = tech

        advance_phase()
        next_action = (
            "Ask a follow-up question." if state.follow_up_pending
            else f"Ask the next Phase {state.current_phase} question."
        )
        push_state(next_action)

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        item = getattr(event, "item", None)
        if not isinstance(item, llm_module.ChatMessage):
            return
        text = item.text_content.strip()
        if not text:
            return
        if item.role == "assistant":
            publish_transcript("AI", text)
            for sentence in text.split("?"):
                q = sentence.strip()
                if q:
                    state.add_question(q + "?")
                    break
            if not state.follow_up_pending:
                state.question_index += 1
            state.follow_up_pending = False

            if state.current_phase == "E" and state.question_index >= 1:
                asyncio.create_task(signal_interview_complete())

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        state_map = {
            "idle": "listening",
            "listening": "listening",
            "thinking": "thinking",
            "speaking": "speaking",
        }
        publish_status(state_map.get(getattr(event, "new_state", ""), "connected"))

    @session.on("error")
    def on_session_error(event):
        detail = str(getattr(event, "error", "Agent session error"))
        logger.error("Agent session error: %s", detail)
        publish_status("error", detail)

    # ── Greeting ──────────────────────────────────────────────────────────────

    if settings_loaded:
        greeting = (
            f"Hello {candidate_name}, I'm Eve, your AI recruiter. "
            f"I've reviewed your resume and I'm excited to chat with you today. "
            f"We'll spend about {duration_minutes} minutes covering your experience, "
            f"technical skills, and a couple of behavioral questions. "
            f"Are you ready to get started?"
        )
    else:
        greeting = (
            f"Hello {candidate_name}, I'm Eve, your AI recruiter. "
            f"Before we begin, could you tell me what role you're interviewing for "
            f"and which technical areas you'd like us to focus on?"
        )

    publish_status("speaking")
    greeting_handle = session.say(greeting, allow_interruptions=True, add_to_chat_ctx=True)
    await greeting_handle.wait_for_playout()
    state.add_question(greeting)
    state.question_index += 1
    publish_status("listening")
    logger.info("Interview started for room %s", ctx.room.name)


# FIX: Python 3.14 - Initialize event loop at module level for spawned processes
if os.getenv("_LIVEKIT_WORKER_PROCESS") == "1":
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


if __name__ == "__main__":
    # Create event loop if it doesn't exist
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    missing = [v for v in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GROQ_API_KEY"] if not os.getenv(v)]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        raise SystemExit(1)

    import socket as _socket
    def free_port(start=8300):
        for p in range(start, start + 100):
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                    s.bind(("", p))
                    return p
            except OSError:
                continue
        return 0

    # FIX: Set env var so spawned worker processes initialize event loops
    os.environ["_LIVEKIT_WORKER_PROCESS"] = "1"
    
    port = int(os.getenv("LIVEKIT_AGENT_PORT", 0)) or free_port()
    logger.info("Starting agent worker on port %s", port)
    cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, port=port))
