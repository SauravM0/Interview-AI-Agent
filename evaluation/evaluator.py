"""
Post-interview evaluator — powered by Groq (free, no credit card).
Replaces Google Gemini (was: import google.generativeai as genai).
Uses Groq's OpenAI-compatible chat API via the groq Python SDK.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from groq import AsyncGroq

logger = logging.getLogger("Evaluator")

# Free Groq models — both available on free tier
EVAL_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are an expert HR recruiter evaluating a candidate interview transcript. "
    "Be objective and specific. Base every score strictly on evidence in the transcript. "
    "Respond only with valid JSON. No markdown fences, no preamble."
)

TECHNICAL_PROMPT = """Analyze technical competency in this interview transcript.

Return ONLY this JSON (no fences, no extra text):
{{
  "score": <0-10 float>,
  "strengths": ["<concrete strength from transcript>"],
  "gaps": ["<concrete gap from transcript>"],
  "topics_covered": ["<topic 1>", "<topic 2>"],
  "reasoning": "<2-3 sentence summary grounded in transcript evidence>"
}}

Transcript:
{transcript}"""

COMMUNICATION_PROMPT = """Analyze communication quality in this interview transcript.

Return ONLY this JSON (no fences, no extra text):
{{
  "score": <0-10 float>,
  "strengths": ["<strength>"],
  "areas_for_improvement": ["<area>"],
  "reasoning": "<2-3 sentence summary>"
}}

Transcript:
{transcript}"""

REPORT_PROMPT = """Write a professional hiring recommendation report in Markdown.

Technical score: {tech_score}/10
Communication score: {comm_score}/10
Overall score: {overall}/10

## Executive Summary
## Key Strengths
## Technical Assessment
## Communication Assessment
## Areas for Development
## Hiring Recommendation
(Choose one: Strong Hire | Hire | Hold | No Hire — explain why in 2 sentences)

Base every statement on specific evidence from this transcript:
{transcript}"""


class InterviewEvaluator:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not set — evaluator disabled. "
                "Get a free key at console.groq.com (no credit card)."
            )
            self._client: AsyncGroq | None = None
        else:
            self._client = AsyncGroq(api_key=api_key)
            logger.info("Evaluator using Groq model: %s", EVAL_MODEL)

    async def evaluate_interview(self, transcript_text: str) -> Dict[str, Any]:
        if not self._client:
            return {
                "error": "GROQ_API_KEY not configured",
                "overall_score": 0,
                "scores": {},
                "report_markdown": "# Evaluation unavailable\n\nGROQ_API_KEY not set.",
            }

        tech = await self._analyze(
            TECHNICAL_PROMPT.format(transcript=transcript_text)
        )
        comm = await self._analyze(
            COMMUNICATION_PROMPT.format(transcript=transcript_text)
        )
        tech_score = tech.get("score", 0)
        comm_score = comm.get("score", 0)
        overall = round((tech_score + comm_score) / 2, 1)

        report = await self._generate_report(transcript_text, tech_score, comm_score, overall)

        return {
            "scores": {"technical": tech, "communication": comm},
            "overall_score": overall,
            "report_markdown": report,
        }

    async def _analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            response = await self._client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            text = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError as exc:
            logger.error("Evaluator JSON parse error: %s", exc)
            return {"score": 0, "reasoning": f"Parse error: {exc}"}
        except Exception as exc:
            logger.error("Evaluator analysis failed: %s", exc)
            return {"score": 0, "reasoning": f"Analysis failed: {exc}"}

    async def _generate_report(
        self, transcript: str, tech_score: float, comm_score: float, overall: float
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": REPORT_PROMPT.format(
                            transcript=transcript,
                            tech_score=tech_score,
                            comm_score=comm_score,
                            overall=overall,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=1200,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)
            return f"# Evaluation Report\n\nGeneration failed: {exc}"
