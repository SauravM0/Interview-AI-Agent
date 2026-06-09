"""
Smoke test — run before every demo to catch issues early.
Usage: python scripts/smoke_test.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv()


async def test_groq_llm() -> bool:
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        resp = await client.chat.completions.create(
            model=os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        result = resp.choices[0].message.content.strip()
        print(f"  [PASS] Groq LLM: {result}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Groq LLM: {exc}")
        return False


async def test_groq_fallback() -> bool:
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        resp = await client.chat.completions.create(
            model=os.getenv("GROQ_LLM_FALLBACK_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        result = resp.choices[0].message.content.strip()
        print(f"  [PASS] Groq fallback LLM: {result}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Groq fallback LLM: {exc}")
        return False


async def test_evaluator() -> bool:
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from evaluation.evaluator import InterviewEvaluator
        ev = InterviewEvaluator()
        result = await ev.evaluate_interview(
            "Interviewer: Tell me about Python. Candidate: I use Python for web APIs with FastAPI."
        )
        score = result.get("overall_score", 0)
        print(f"  [PASS] Evaluator: overall_score={score}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Evaluator: {exc}")
        return False


async def test_api_health() -> bool:
    try:
        import httpx
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{api_url}/health")
        if resp.status_code == 200:
            print(f"  [PASS] API server: {resp.json()}")
            return True
        print(f"  [FAIL] API server: status {resp.status_code}")
        return False
    except Exception as exc:
        print(f"  [FAIL] API server: {exc}")
        return False


def test_no_gemini_imports() -> bool:
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "google.generativeai", "--include=*.py", "."],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"  [FAIL] Gemini imports still present:\n{result.stdout}")
        return False
    print("  [PASS] No Gemini imports found")
    return True


async def main():
    print("\n=== Interview AI — Pre-Demo Smoke Test ===\n")

    results = await asyncio.gather(
        test_groq_llm(),
        test_groq_fallback(),
        test_evaluator(),
        test_api_health(),
        return_exceptions=True,
    )

    gemini_clean = test_no_gemini_imports()
    all_passed = all(r is True for r in results) and gemini_clean

    print("\n" + ("All checks passed — ready to demo!" if all_passed
                  else "Some checks failed — fix before demoing."))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
