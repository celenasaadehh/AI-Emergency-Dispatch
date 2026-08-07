import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ENV_FILE = Path(__file__).with_name(".env.local")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = ["gemini-2.5-flash-lite"]
DEFAULT_CONTEXT_MD = "gemini_incident_context.md"
DEFAULT_REPORT_MD = "incident_report.md"


SYSTEM_INSTRUCTION = """
You are an emergency dispatch decision-support analyst.

Use the incident context to produce a concise operational report for firefighters,
EMS, and police dispatchers. Treat the nearby places as evidence, not certainty.
Do not invent facts that are not supported by the incident context. If the data is
ambiguous, say so explicitly.

Your report must include:
- Severity assessment: low, medium, high, or critical.
- Likely incident type.
- What is likely burning or affected.
- Potential escalation paths.
- Nearby risk factors and vulnerable places.
- Unit recommendations, including fire units, ambulances, police, and any special
  units such as hazmat, evacuation support, rescue, traffic control, or utility crew.
- Required fire equipment types using only these labels when applicable:
  Type A, Type B, Type C, Type D, Type K.
- Confidence level and data gaps.

Return Markdown only. Do not wrap the answer in a code block.
""".strip()


REPORT_PROMPT = """
Create an emergency incident report from the context below.

Use this structure:

# Emergency Incident Report

## Incident Summary

## Severity Assessment

## Likely Type and Burning Material

## Nearby Risks

## Potential Escalation

## Unit Recommendation

Include a compact table with:
| Unit Type | Recommendation | Reason |

## Required Equipment Types

List any needed equipment labels from this exact set only: Type A, Type B,
Type C, Type D, Type K. If uncertain, say which type is the conservative
default and why.

## Confidence and Data Gaps

Context:

{context}
""".strip()


def load_local_env(env_file: Path = ENV_FILE) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_response_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_json}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Gemini returned no text parts: {response_json}")

    return "\n".join(text_parts).strip()


def call_gemini(
    context: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": REPORT_PROMPT.format(context=context)}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    retry_statuses = {429, 500, 502, 503, 504}
    last_error: Optional[Exception] = None
    last_response_text: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if response.status_code in retry_statuses and attempt < max_retries:
                last_response_text = response.text
                wait_seconds = min(60, 5 * attempt)
                print(
                    f"Gemini model {model} returned {response.status_code}. "
                    f"Retrying in {wait_seconds}s ({attempt}/{max_retries})..."
                )
                time.sleep(wait_seconds)
                continue

            last_response_text = response.text
            response.raise_for_status()
            return extract_response_text(response.json())
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = min(60, 5 * attempt)
            print(
                f"Gemini request failed. "
                f"Retrying in {wait_seconds}s ({attempt}/{max_retries})..."
            )
            time.sleep(wait_seconds)

    detail = f" Last response: {last_response_text}" if last_response_text else ""
    raise RuntimeError(
        f"Gemini model {model} failed after {max_retries} attempts.{detail}"
    ) from last_error


def call_gemini_with_fallbacks(
    context: str,
    models: List[str],
    api_key: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
) -> str:
    errors = []
    for model in models:
        try:
            print(f"Calling Gemini model: {model}")
            return call_gemini(
                context=context,
                model=model,
                api_key=api_key,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
            print(f"{model} failed. Trying next model if available.")

    raise RuntimeError("All Gemini models failed:\n" + "\n\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Gemini with a prepared incident context and write a Markdown report."
    )
    parser.add_argument("--context", default=DEFAULT_CONTEXT_MD)
    parser.add_argument("--output", default=DEFAULT_REPORT_MD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--fallback-model",
        action="append",
        default=DEFAULT_FALLBACK_MODELS,
        help="Fallback model to try if --model fails. Can be passed multiple times.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()

    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key or api_key == "put_your_gemini_api_key_here":
        raise RuntimeError(
            f"Set {GEMINI_API_KEY_ENV} in {ENV_FILE} before calling Gemini."
        )

    context_path = Path(args.context)
    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")

    context = context_path.read_text(encoding="utf-8")
    models = [args.model] + [
        fallback for fallback in args.fallback_model if fallback != args.model
    ]
    report = call_gemini_with_fallbacks(
        context=context,
        models=models,
        api_key=api_key,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(f"Saved Gemini report to {args.output}")


if __name__ == "__main__":
    main()
