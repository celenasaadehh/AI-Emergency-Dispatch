import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "pipeline_output" / "n8n_runs"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PLACES_SOURCE = "osm"
DEFAULT_SKIP_LLM = False
DEFAULT_RADIUS_M = 100
DEFAULT_TIMEOUT_SECONDS = 420

VALID_PLACES_SOURCES = {"osm", "google", "both"}


class ApiError(Exception):
    def __init__(self, status: int, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}


def load_local_env(path: Path = PROJECT_ROOT / ".env.local") -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return coerce_bool(value, default)


def coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def first_present(data: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return None


def nested_dict(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def parse_json_body(raw_body: bytes) -> Dict[str, Any]:
    if not raw_body:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be JSON.")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "Request body is not valid JSON.",
            {"jsonError": str(exc)},
        ) from exc

    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Request JSON must be an object.")
    return payload


def unwrap_n8n_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    body = payload.get("body")
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return payload
        return parsed if isinstance(parsed, dict) else payload
    return payload


def extract_whatsapp_message(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        if isinstance(message, dict):
            return message
    except (KeyError, IndexError, TypeError):
        pass

    message = data.get("message")
    if isinstance(message, dict):
        return message

    messages = data.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        return messages[0]

    return {}


def coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"Missing or invalid {field_name}. Send numeric latitude and longitude.",
        ) from exc


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def safe_run_id(value: Any) -> str:
    text = str(value or "incident").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-")[:80] or "incident"


def message_text(data: Dict[str, Any], wa_message: Dict[str, Any]) -> str:
    text_payload = nested_dict(wa_message, "text")
    return str(
        first_present(data, ("messageText", "message_text", "text", "description"))
        or text_payload.get("body")
        or wa_message.get("body")
        or ""
    )


def normalize_incident(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = unwrap_n8n_body(payload)
    wa_message = extract_whatsapp_message(data)
    data_location = nested_dict(data, "location")
    message_location = nested_dict(wa_message, "location")

    latitude = first_present(
        data,
        ("latitude", "lat", "report_latitude", "seed_lat"),
    )
    longitude = first_present(
        data,
        ("longitude", "lon", "lng", "report_longitude", "seed_lon"),
    )
    latitude = latitude if latitude is not None else first_present(
        message_location or data_location,
        ("latitude", "lat"),
    )
    longitude = longitude if longitude is not None else first_present(
        message_location or data_location,
        ("longitude", "lon", "lng"),
    )

    incident_id = (
        first_present(data, ("incidentId", "incident_id", "id"))
        or wa_message.get("id")
        or f"WA_{int(time.time())}"
    )

    required_equipment = first_present(
        data,
        (
            "requiredEquipment",
            "required_equipment",
            "requiredEquipmentTypes",
            "required_equipment_types",
            "equipment",
        ),
    )

    return {
        "incidentId": str(incident_id),
        "eventType": str(first_present(data, ("eventType", "event_type")) or "FIRE"),
        "latitude": coerce_float(latitude, "latitude"),
        "longitude": coerce_float(longitude, "longitude"),
        "severity": first_present(data, ("severity", "severity_score")) or 80,
        "radius_m": first_present(data, ("radius_m", "radius", "nearbyRadiusM"))
        or DEFAULT_RADIUS_M,
        "messageText": message_text(data, wa_message),
        "from": first_present(data, ("from", "phone", "sender")) or wa_message.get("from") or "",
        "requiredEquipment": required_equipment or "",
        "fire_class": first_present(data, ("fire_class", "fireClass", "true_class")) or "",
    }


def request_option(data: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    unwrapped = unwrap_n8n_body(data)
    value = first_present(unwrapped, names)
    if value is not None:
        return value
    return first_present(data, names) or default


def write_incident_csv(path: Path, incident: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "incidentId",
        "eventType",
        "latitude",
        "longitude",
        "severity",
        "radius_m",
        "messageText",
        "from",
        "requiredEquipment",
        "fire_class",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: csv_value(incident.get(name)) for name in fieldnames})


def build_pipeline_command(
    payload: Dict[str, Any],
    incident: Dict[str, Any],
    input_csv: Path,
    output_dir: Path,
) -> Tuple[Sequence[str], int]:
    places_source = str(
        request_option(
            payload,
            ("placesSource", "places_source"),
            os.getenv("PIPELINE_PLACES_SOURCE", DEFAULT_PLACES_SOURCE),
        )
    ).lower()
    if places_source not in VALID_PLACES_SOURCES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "Invalid placesSource. Use one of: osm, google, both.",
        )

    skip_llm = coerce_bool(
        request_option(payload, ("skipLlm", "skipLLM", "skip_llm"), None),
        env_bool("PIPELINE_SKIP_LLM", DEFAULT_SKIP_LLM),
    )
    timeout_seconds = int(
        request_option(
            payload,
            ("pipelineTimeoutSeconds", "timeoutSeconds", "timeout_seconds"),
            os.getenv("PIPELINE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        )
    )

    command = [
        sys.executable,
        "run_incident_pipeline.py",
        "--input",
        str(input_csv),
        "--incident-id",
        incident["incidentId"],
        "--output-dir",
        str(output_dir),
        "--radius-col",
        "radius_m",
        "--places-source",
        places_source,
    ]

    if skip_llm:
        command.append("--skip-llm")

    required_equipment = incident.get("requiredEquipment")
    if required_equipment:
        command.extend(["--required-equipment", csv_value(required_equipment)])

    model = request_option(payload, ("model",), None)
    if model:
        command.extend(["--model", str(model)])

    return command, timeout_seconds


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    incident = normalize_incident(payload)
    run_id = f"{safe_run_id(incident['incidentId'])}_{int(time.time())}"
    output_dir = RUNS_DIR / run_id
    input_csv = output_dir / "incident_from_n8n.csv"
    compact_output = output_dir / "dispatch_recommendations_frontend.json"

    write_incident_csv(input_csv, incident)
    command, timeout_seconds = build_pipeline_command(payload, incident, input_csv, output_dir)

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ApiError(
            HTTPStatus.GATEWAY_TIMEOUT,
            "Pipeline timed out.",
            {
                "timeoutSeconds": timeout_seconds,
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            },
        ) from exc

    if completed.returncode != 0:
        raise ApiError(
            HTTPStatus.BAD_GATEWAY,
            "Pipeline failed.",
            {
                "returnCode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "outputDir": str(output_dir.relative_to(PROJECT_ROOT)),
            },
        )

    if not compact_output.exists():
        raise ApiError(
            HTTPStatus.BAD_GATEWAY,
            "Pipeline finished but compact dispatch output was not created.",
            {"outputPath": str(compact_output.relative_to(PROJECT_ROOT))},
        )

    try:
        result = json.loads(compact_output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError(
            HTTPStatus.BAD_GATEWAY,
            "Pipeline output is not valid JSON.",
            {"outputPath": str(compact_output.relative_to(PROJECT_ROOT))},
        ) from exc

    if not isinstance(result, dict):
        raise ApiError(
            HTTPStatus.BAD_GATEWAY,
            "Pipeline output JSON must be an object.",
            {"outputPath": str(compact_output.relative_to(PROJECT_ROOT))},
        )

    return result


class DispatchApiHandler(BaseHTTPRequestHandler):
    server_version = "EmergencyDispatchApi/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
            return
        self.send_json(HTTPStatus.OK, {"ok": True, "service": "emergency-dispatch-api"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/process-incident":
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = parse_json_body(self.rfile.read(length))
            result = run_pipeline(payload)
            self.send_json(HTTPStatus.OK, result)
        except ApiError as exc:
            self.send_json(
                exc.status,
                {
                    "ok": False,
                    "error": exc.message,
                    "details": exc.details,
                },
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "ok": False,
                    "error": "Unexpected server error.",
                    "details": {"type": type(exc).__name__, "message": str(exc)},
                },
            )

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Iterable[Any]) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP bridge for n8n/WhatsApp JSON into the emergency dispatch pipeline."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), DispatchApiHandler)
    print(f"Emergency dispatch API listening on http://{args.host}:{args.port}")
    print("POST incident JSON to /process-incident")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
