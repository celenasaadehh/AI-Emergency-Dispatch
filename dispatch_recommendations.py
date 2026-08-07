import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FIRE_STATIONS_JSON = "data/fire_stations.json"
DEFAULT_FULL_OUTPUT_JSON = "pipeline_output/dispatch_recommendations.json"
DEFAULT_COMPACT_OUTPUT_JSON = "pipeline_output/dispatch_recommendations_frontend.json"

EQUIPMENT_ORDER = ["Type A", "Type B", "Type C", "Type D", "Type K"]

EQUIPMENT_DEFINITIONS = {
    "Type A": "Ordinary combustibles such as wood, paper, vegetation, textiles, and trash.",
    "Type B": "Flammable liquids and gases such as fuel, oil, gasoline, diesel, and solvents.",
    "Type C": "Energized electrical equipment, wiring, transformers, and charging infrastructure.",
    "Type D": "Combustible metals and specialist metal-fire response equipment.",
    "Type K": "Cooking oils, grease, and commercial kitchen fire response equipment.",
}

DIRECT_EQUIPMENT_COLUMNS = (
    "requiredEquipment",
    "required_equipment",
    "requiredEquipmentTypes",
    "required_equipment_types",
    "equipmentTypes",
    "equipment_types",
    "equipment",
    "fireClass",
    "fire_class",
    "true_class",
)

KEYWORD_RULES = (
    ("Type K", ("cooking oil", "grease", "oil flare", "kitchen", "commercial kitchen")),
    (
        "Type B",
        (
            "fuel",
            "gasoline",
            "diesel",
            "flammable liquid",
            "flammable gas",
            "solvent",
            "oil smell",
            "fuel station",
            "gas station",
            "petrol",
            "chemical",
            "hazmat",
        ),
    ),
    (
        "Type C",
        (
            "electrical",
            "electric",
            "short circuit",
            "wires",
            "wiring",
            "charging_station",
            "charging station",
            "substation",
            "transformer",
            "power",
            "utility crew",
        ),
    ),
    (
        "Type D",
        (
            "combustible metal",
            "metal fire",
            "magnesium",
            "titanium",
            "sodium",
            "potassium metal",
            "lithium metal",
        ),
    ),
    (
        "Type A",
        (
            "wood",
            "paper",
            "vegetation",
            "forest",
            "textile",
            "clothing",
            "trash",
            "ordinary combustibles",
            "furniture",
            "building",
            "warehouse",
            "commercial",
            "residential",
        ),
    ),
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return radius_km * 2 * atan2(sqrt(a), sqrt(1 - a))


def estimate_eta_minutes(distance_km: float) -> int:
    speed_kmh = 50
    return int(round((distance_km / speed_kmh) * 60))


def normalize_equipment_type(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    upper = text.upper().replace("_", " ").replace("-", " ")
    match = re.search(r"\b(?:TYPE|CLASS)?\s*([ABCDK])\b", upper)
    if not match:
        return None

    normalized = f"Type {match.group(1)}"
    return normalized if normalized in EQUIPMENT_DEFINITIONS else None


def ordered_unique_equipment(values: Iterable[Any]) -> List[str]:
    found = set()
    for value in values:
        normalized = normalize_equipment_type(value)
        if normalized:
            found.add(normalized)
    return [equipment for equipment in EQUIPMENT_ORDER if equipment in found]


def parse_equipment_value(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return ordered_unique_equipment(value)

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return ordered_unique_equipment(parsed)

    tokens = re.split(r"[,;/|]+", text)
    return ordered_unique_equipment(tokens if len(tokens) > 1 else [text])


def read_text_if_exists(path: Optional[Path]) -> str:
    if not path:
        return ""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return value


def parse_context_json(context_text: str) -> Dict[str, Any]:
    if not context_text:
        return {}

    match = re.search(r"```json\s*(\{.*?\})\s*```", context_text, flags=re.S)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def normalize_severity(value: Any) -> Optional[str]:
    safe_value = json_safe_value(value)
    if safe_value is None:
        return None

    text = str(safe_value).strip().lower()
    if text in {"low", "medium", "high", "critical"}:
        return text

    try:
        score = float(text)
    except ValueError:
        return None

    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def extract_severity_from_report(report_text: str) -> Optional[str]:
    if not report_text:
        return None

    section_match = re.search(
        r"##\s*Severity Assessment(?P<section>.*?)(?:\n##\s+|\Z)",
        report_text,
        flags=re.S | re.I,
    )
    search_text = section_match.group("section") if section_match else report_text
    match = re.search(r"\b(low|medium|high|critical)\b", search_text, flags=re.I)
    return match.group(1).lower() if match else None


def closest_context_place(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    places = context.get("closest_recognizable_places")
    if not isinstance(places, list):
        return None
    return next((place for place in places if isinstance(place, dict)), None)


def build_incident_summary(
    incident: Dict[str, Any],
    incident_latitude: float,
    incident_longitude: float,
    context: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    event_type = str(incident.get("eventType") or incident.get("event_type") or "incident")
    closest_place = closest_context_place(context)
    location_name = None
    distance_m = None

    if closest_place:
        location_name = json_safe_value(closest_place.get("name"))
        distance_m = json_safe_value(closest_place.get("distance_m"))

    severity_label = (
        extract_severity_from_report(report_text)
        or normalize_severity(incident.get("severity"))
    )
    raw_severity = json_safe_value(incident.get("severity"))

    if location_name and distance_m is not None:
        description = (
            f"{event_type.title()} reported near {location_name}, about "
            f"{round(float(distance_m), 1)} m from the incident coordinates."
        )
    elif location_name:
        description = f"{event_type.title()} reported near {location_name}."
    else:
        description = (
            f"{event_type.title()} reported at coordinates "
            f"{incident_latitude:.5f}, {incident_longitude:.5f}."
        )

    return {
        "locationName": location_name,
        "description": description,
        "severity": raw_severity,
        "severityLabel": severity_label,
    }


def load_incident_from_csv(input_csv: Path, incident_id: Optional[str]) -> Dict[str, Any]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    selected = None
    if incident_id:
        selected = next(
            (row for row in rows if str(row.get("incidentId")) == str(incident_id)),
            None,
        )
        if selected is None:
            raise ValueError(f"Incident {incident_id} was not found in {input_csv}")
    else:
        selected = rows[0]

    return dict(selected)


def incident_coordinate(incident: Dict[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = incident.get(name)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Could not find coordinate column from {', '.join(names)}")


def direct_equipment_from_incident(incident: Dict[str, Any]) -> List[str]:
    for column in DIRECT_EQUIPMENT_COLUMNS:
        equipment = parse_equipment_value(incident.get(column))
        if equipment:
            return equipment
    return []


def explicit_equipment_from_report(report_text: str) -> List[str]:
    matches = re.findall(r"\b(?:Type|Class)\s*[- ]?([ABCDK])\b", report_text, flags=re.I)
    return ordered_unique_equipment(f"Type {match}" for match in matches)


def heuristic_equipment_from_text(text: str) -> List[str]:
    lowered = text.lower()
    matches = []
    for equipment, keywords in KEYWORD_RULES:
        if any(keyword in lowered for keyword in keywords):
            matches.append(equipment)
    return ordered_unique_equipment(matches)


def determine_required_equipment(
    incident: Dict[str, Any],
    report_text: str = "",
    context_text: str = "",
) -> Tuple[List[str], str]:
    if incident.get("__requiredEquipmentOverride"):
        override = parse_equipment_value(incident.get("__requiredEquipmentOverride"))
        if override:
            return override, "manual_override"

    direct = direct_equipment_from_incident(incident)
    if direct:
        return direct, "incident_csv"

    explicit = explicit_equipment_from_report(report_text)
    if explicit:
        return explicit, "llm_report_explicit_type_mentions"

    if report_text:
        inferred = heuristic_equipment_from_text(report_text)
        if inferred:
            return inferred, "heuristic_from_llm_report"

    incident_text = " ".join(str(value or "") for value in incident.values())
    inferred = heuristic_equipment_from_text(" ".join([context_text, incident_text]))
    if inferred:
        return inferred, "heuristic_from_context_and_incident"

    if str(incident.get("eventType") or "").strip().upper() == "FIRE":
        return ["Type A"], "fallback_fire_default"

    return [], "unknown"


def load_fire_stations(path: Path) -> List[Dict[str, Any]]:
    stations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stations, list):
        raise ValueError(f"{path} must contain a JSON array")

    fire_stations = [
        station
        for station in stations
        if str(station.get("stationType") or "FIRE").upper() == "FIRE"
    ]
    if not fire_stations:
        raise ValueError(f"No FIRE stations found in {path}")
    return fire_stations


def rank_stations(
    stations: Sequence[Dict[str, Any]],
    incident_latitude: float,
    incident_longitude: float,
    required_equipment: Sequence[str],
) -> List[Dict[str, Any]]:
    ranked = []
    required_set = set(required_equipment)
    for station in stations:
        distance_km = haversine_km(
            incident_latitude,
            incident_longitude,
            float(station["latitude"]),
            float(station["longitude"]),
        )
        station_equipment = ordered_unique_equipment(station.get("equipment") or [])
        station_equipment_set = set(station_equipment)
        missing = [item for item in required_equipment if item not in station_equipment_set]
        ranked.append(
            {
                "stationId": station.get("stationId"),
                "name": station.get("name"),
                "region": station.get("region"),
                "stationType": station.get("stationType", "FIRE"),
                "latitude": station.get("latitude"),
                "longitude": station.get("longitude"),
                "phone": station.get("phone", ""),
                "distanceKm": round(distance_km, 2),
                "etaMinutes": estimate_eta_minutes(distance_km),
                "equipment": station_equipment,
                "hasRequiredEquipment": required_set.issubset(station_equipment_set),
                "missingEquipment": missing,
                "equipmentIsDummy": bool(station.get("equipmentIsDummy", True)),
                "locationSource": station.get("locationSource"),
                "notes": station.get("notes"),
            }
        )

    ranked.sort(key=lambda item: item["distanceKm"])
    return ranked


def add_distance_delta(
    station: Optional[Dict[str, Any]],
    reference_distance_km: Optional[float],
) -> Optional[Dict[str, Any]]:
    if station is None:
        return None
    result = dict(station)
    if reference_distance_km is None:
        result["distanceDeltaKm"] = None
    else:
        result["distanceDeltaKm"] = round(result["distanceKm"] - reference_distance_km, 2)
    return result


def build_dispatch_recommendations(
    incident: Dict[str, Any],
    stations: Sequence[Dict[str, Any]],
    report_text: str = "",
    context_text: str = "",
) -> Dict[str, Any]:
    incident_latitude = incident_coordinate(incident, ("latitude", "lat", "report_latitude"))
    incident_longitude = incident_coordinate(incident, ("longitude", "lon", "lng", "report_longitude"))
    context = parse_context_json(context_text)
    incident_summary = build_incident_summary(
        incident=incident,
        incident_latitude=incident_latitude,
        incident_longitude=incident_longitude,
        context=context,
        report_text=report_text,
    )
    required_equipment, equipment_source = determine_required_equipment(
        incident,
        report_text=report_text,
        context_text=context_text,
    )
    ranked = rank_stations(
        stations,
        incident_latitude,
        incident_longitude,
        required_equipment,
    )

    equipped = [station for station in ranked if station["hasRequiredEquipment"]]
    not_equipped = [station for station in ranked if not station["hasRequiredEquipment"]]
    closest_with = equipped[0] if equipped else None
    additional_with = equipped[1] if len(equipped) > 1 else None
    closest_without = not_equipped[0] if not_equipped else None
    reference_distance = closest_with["distanceKm"] if closest_with else None

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "incident": {
            "incidentId": incident.get("incidentId") or incident.get("incident_id"),
            "eventType": incident.get("eventType") or incident.get("event_type"),
            "latitude": incident_latitude,
            "longitude": incident_longitude,
            "severity": incident_summary["severity"],
            "severityLabel": incident_summary["severityLabel"],
            "locationName": incident_summary["locationName"],
            "description": incident_summary["description"],
        },
        "report": report_text,
        "requiredEquipment": required_equipment,
        "requiredEquipmentSource": equipment_source,
        "equipmentDefinitions": EQUIPMENT_DEFINITIONS,
        "recommendations": {
            "closestWithRequiredEquipment": add_distance_delta(closest_with, reference_distance),
            "closestWithoutRequiredEquipment": add_distance_delta(closest_without, reference_distance),
            "additionalWithRequiredEquipment": add_distance_delta(additional_with, reference_distance),
        },
        "rankedStations": [
            add_distance_delta(station, reference_distance) for station in ranked
        ],
        "notes": [
            "Station distance is straight-line haversine distance, not road travel distance.",
            "Location data is mixed sourced and dummy city-level data; equipment assignments are dummy data for testing.",
            "The LLM may help identify required equipment, but station matching is deterministic code.",
        ],
    }


def unit_equipment(
    station: Optional[Dict[str, Any]],
    preferred_equipment: Sequence[str],
    limit: int,
) -> List[str]:
    if station is None:
        return []

    station_equipment = ordered_unique_equipment(station.get("equipment") or [])
    station_equipment_set = set(station_equipment)
    selected = [
        equipment
        for equipment in preferred_equipment
        if equipment in station_equipment_set
    ]
    selected.extend(equipment for equipment in station_equipment if equipment not in selected)
    return selected[:limit]


def compact_truck_assignments(
    recommendations: Dict[str, Optional[Dict[str, Any]]],
    required_equipment: Sequence[str],
) -> List[Dict[str, Any]]:
    primary_station = (
        recommendations["closestWithRequiredEquipment"]
        or recommendations["closestWithoutRequiredEquipment"]
        or recommendations["additionalWithRequiredEquipment"]
    )
    additional_station = recommendations["additionalWithRequiredEquipment"]
    trucks: List[Dict[str, Any]] = []

    if primary_station:
        trucks.append(
            {
                "stationId": primary_station.get("stationId"),
                "unitLabel": "Unit 1",
                "mission": "Primary fire suppression",
                "equipment": unit_equipment(primary_station, required_equipment, 3),
            }
        )
        support_preferred = required_equipment[2:] if len(required_equipment) > 2 else required_equipment
        trucks.append(
            {
                "stationId": primary_station.get("stationId"),
                "unitLabel": "Unit 2",
                "mission": "Search and rescue support",
                "equipment": unit_equipment(primary_station, support_preferred, 2),
            }
        )

    if additional_station:
        trucks.append(
            {
                "stationId": additional_station.get("stationId"),
                "unitLabel": f"Unit {len(trucks) + 1}",
                "mission": "Water supply and perimeter control",
                "equipment": unit_equipment(
                    additional_station,
                    ["Type A", "Type D", "Type B", "Type C", "Type K"],
                    2,
                ),
            }
        )

    return trucks


def compact_frontend_payload(full_payload: Dict[str, Any]) -> Dict[str, Any]:
    recommendations = full_payload["recommendations"]
    cards = [
        ("closest_with_required_equipment", recommendations["closestWithRequiredEquipment"]),
        ("closest_without_required_equipment", recommendations["closestWithoutRequiredEquipment"]),
        ("additional_with_required_equipment", recommendations["additionalWithRequiredEquipment"]),
    ]

    return {
        "generatedAt": full_payload["generatedAt"],
        "incident": full_payload["incident"],
        "requiredEquipment": full_payload["requiredEquipment"],
        "requiredEquipmentSource": full_payload["requiredEquipmentSource"],
        "trucks": compact_truck_assignments(
            recommendations,
            full_payload["requiredEquipment"],
        ),
        "stationOptions": [
            {
                "role": role,
                "stationId": station.get("stationId"),
                "name": station.get("name"),
                "region": station.get("region"),
                "distanceKm": station.get("distanceKm"),
                "distanceDeltaKm": station.get("distanceDeltaKm"),
                "etaMinutes": station.get("etaMinutes"),
                "equipment": station.get("equipment"),
                "hasRequiredEquipment": station.get("hasRequiredEquipment"),
                "missingEquipment": station.get("missingEquipment"),
                "phone": station.get("phone"),
            }
            for role, station in cards
            if station is not None
        ],
        "markdownReport": full_payload.get("report", ""),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dispatch_outputs(
    incident: Dict[str, Any],
    stations_path: Path,
    full_output_path: Path,
    compact_output_path: Path,
    report_text: str = "",
    context_text: str = "",
) -> Dict[str, Any]:
    stations = load_fire_stations(stations_path)
    full_payload = build_dispatch_recommendations(
        incident=incident,
        stations=stations,
        report_text=report_text,
        context_text=context_text,
    )
    write_json(full_output_path, full_payload)
    write_json(compact_output_path, compact_frontend_payload(full_payload))
    return full_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank fire stations by distance and required fire-equipment type."
    )
    parser.add_argument("--input", required=True, help="Incident CSV path")
    parser.add_argument("--incident-id", help="Incident ID to process. Defaults to first row.")
    parser.add_argument("--stations", default=DEFAULT_FIRE_STATIONS_JSON)
    parser.add_argument("--report", help="Optional LLM report Markdown path")
    parser.add_argument("--context", help="Optional LLM context Markdown path")
    parser.add_argument(
        "--required-equipment",
        action="append",
        help=(
            "Override required equipment. Can be repeated or comma-separated, "
            "for example: --required-equipment 'Type A,Type C'."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_FULL_OUTPUT_JSON)
    parser.add_argument("--compact-output", default=DEFAULT_COMPACT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    incident = load_incident_from_csv(Path(args.input), args.incident_id)
    if args.required_equipment:
        incident["__requiredEquipmentOverride"] = ",".join(args.required_equipment)
    payload = write_dispatch_outputs(
        incident=incident,
        stations_path=Path(args.stations),
        full_output_path=Path(args.output),
        compact_output_path=Path(args.compact_output),
        report_text=read_text_if_exists(Path(args.report)) if args.report else "",
        context_text=read_text_if_exists(Path(args.context)) if args.context else "",
    )
    print(f"Saved dispatch recommendations to {args.output}")
    print(f"Saved compact front-end recommendations to {args.compact_output}")
    print(f"Required equipment: {', '.join(payload['requiredEquipment']) or 'unknown'}")


if __name__ == "__main__":
    main()
