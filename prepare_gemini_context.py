import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_NEARBY_CSV = "incident_nearby_places_structured.csv"
DEFAULT_INCIDENTS_CSV = "incidents.csv"
DEFAULT_OUTPUT_MD = "gemini_incident_context.md"

RISK_KEYWORDS = (
    "fuel",
    "gas",
    "hospital",
    "pharmacy",
    "clinic",
    "doctor",
    "school",
    "university",
    "college",
    "police",
    "fire",
    "checkpoint",
    "power",
    "substation",
    "transformer",
    "industrial",
    "warehouse",
    "factory",
    "restaurant",
    "cafe",
    "fast_food",
    "bakery",
    "hotel",
    "lodging",
    "church",
    "mosque",
    "place_of_worship",
    "government",
    "diplomatic",
    "office",
    "bridge",
    "parking",
)

GENERIC_UNNAMED_TYPES = {
    "building:yes",
    "other",
}

LAT_COLS = ("latitude", "lat", "report_latitude", "seed_lat")
LON_COLS = ("longitude", "lon", "lng", "report_longitude", "seed_lon")


def get_incident_coords(incident: pd.Series):
    """Extract lat/lon from an incident row, tolerating alternate column names."""
    lat = lon = None
    for col in LAT_COLS:
        if col in incident.index and pd.notna(incident.get(col)):
            lat = float(incident[col])
            break
    for col in LON_COLS:
        if col in incident.index and pd.notna(incident.get(col)):
            lon = float(incident[col])
            break
    return lat, lon


def load_incident_row(incidents_csv: str, incident_id: Optional[str]) -> pd.Series:
    incidents = pd.read_csv(incidents_csv)
    if incident_id:
        matches = incidents[incidents["incidentId"].astype(str) == str(incident_id)]
        if matches.empty:
            raise ValueError(f"Incident {incident_id} was not found in {incidents_csv}")
        return matches.iloc[0]

    first_valid = incidents[incidents["incidentId"].notna()]
    if first_valid.empty:
        raise ValueError(f"No valid incidentId found in {incidents_csv}")
    return first_valid.iloc[0]


def is_unnamed(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("unnamed")


def is_generic_unnamed(row: pd.Series) -> bool:
    if not is_unnamed(row.get("place_name")):
        return False
    return str(row.get("place_type") or "").strip().lower() in GENERIC_UNNAMED_TYPES


def risk_score(row: pd.Series) -> int:
    text = " ".join(
        str(row.get(col) or "").lower()
        for col in ("place_name", "place_type", "place_role", "additional_info", "address")
    )
    return sum(1 for keyword in RISK_KEYWORDS if keyword in text)


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return json_safe_value(value.item())
        except (TypeError, ValueError):
            pass

    return value


def row_to_place(row: pd.Series) -> Dict[str, Any]:
    place = {
        "name": json_safe_value(row.get("place_name")),
        "source": json_safe_value(row.get("source")),
        "distance_m": json_safe_value(row.get("distance_to_fire_m")),
        "type": json_safe_value(row.get("place_type")),
        "role": json_safe_value(row.get("place_role")),
        "address": json_safe_value(row.get("address")),
        "lat": json_safe_value(row.get("place_latitude")),
        "lon": json_safe_value(row.get("place_longitude")),
    }
    if not pd.isna(row.get("google_place_id")):
        place["google_place_id"] = json_safe_value(row.get("google_place_id"))
    if not pd.isna(row.get("osm_type")) and not pd.isna(row.get("osm_id")):
        place["osm_ref"] = f"{row.get('osm_type')}/{row.get('osm_id')}"
    return place


def summarize_generic_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "generic_unnamed_count": int(len(rows)),
        "by_type": {},
        "within_50m": 0,
        "within_100m": 0,
        "within_200m": 0,
        "closest_distance_m": None,
    }
    if rows.empty:
        return summary

    distances = pd.to_numeric(rows["distance_to_fire_m"], errors="coerce").dropna()
    summary["by_type"] = {
        str(key): int(value)
        for key, value in rows["place_type"].fillna("unknown").value_counts().to_dict().items()
    }
    summary["within_50m"] = int((distances <= 50).sum())
    summary["within_100m"] = int((distances <= 100).sum())
    summary["within_200m"] = int((distances <= 200).sum())
    if not distances.empty:
        summary["closest_distance_m"] = round(float(distances.min()), 2)
    return summary


def context_radius_m(incident: pd.Series, nearby_rows: pd.DataFrame) -> Any:
    if "matched_within_radius_m" in nearby_rows.columns:
        radii = pd.to_numeric(nearby_rows["matched_within_radius_m"], errors="coerce").dropna()
        if not radii.empty:
            return int(radii.max())

    radius = json_safe_value(incident.get("radius_m"))
    if radius is None:
        return None

    try:
        return int(float(radius))
    except (TypeError, ValueError):
        return radius


def build_markdown(
    incident: pd.Series,
    nearby_rows: pd.DataFrame,
    max_places: int,
    weather: Optional[Dict[str, Any]] = None,
) -> str:
    if nearby_rows.empty:
        generic_rows = nearby_rows.copy()
        closest_places = nearby_rows.copy()
        risk_places = nearby_rows.copy()
    else:
        generic_mask = nearby_rows.apply(is_generic_unnamed, axis=1).astype(bool)
        generic_rows = nearby_rows[generic_mask]
        useful_rows = nearby_rows[~generic_mask].copy()
        useful_rows["risk_score"] = useful_rows.apply(risk_score, axis=1)
        useful_rows = useful_rows.sort_values(
            by=["risk_score", "distance_to_fire_m"],
            ascending=[False, True],
        )

        closest_places = useful_rows.sort_values("distance_to_fire_m").head(max_places)
        risk_places = useful_rows[useful_rows["risk_score"] > 0].head(max_places)

    context = {
        "incident": {
            "id": json_safe_value(incident.get("incidentId")),
            "reported_event_type": json_safe_value(incident.get("eventType")),
            "latitude": json_safe_value(incident.get("latitude")),
            "longitude": json_safe_value(incident.get("longitude")),
            "reported_severity": json_safe_value(incident.get("severity")),
            "caller_observations": json_safe_value(incident.get("messageText")),
        },
        "nearby_context_radius_m": context_radius_m(incident, nearby_rows),
        "weather_conditions": weather,
        "generic_unnamed_osm_summary": summarize_generic_rows(generic_rows),
        "closest_recognizable_places": [row_to_place(row) for _, row in closest_places.iterrows()],
        "risk_relevant_places": [row_to_place(row) for _, row in risk_places.iterrows()],
        "required_llm_output": {
            "severity": "low | medium | high | critical",
            "incident_type": "residential_fire | commercial_fire | industrial_fire | fuel_fire | electrical_fire | vehicle_fire | explosion | earthquake | unknown",
            "what_is_likely_burning": "short explanation based on incident and nearby context",
            "potential_escalation": "short escalation assessment",
            "wind_driven_spread": "expected spread direction and speed given wind conditions, and which nearby places lie in that path",
            "nearby_risks": ["list of specific nearby places or categories that matter"],
            "recommended_response": {
                "fire_units": "integer",
                "ambulances": "integer",
                "police_required": "boolean",
                "special_units": ["hazmat", "rescue", "evacuation_support"],
            },
            "required_equipment_types": [
                "Type A | Type B | Type C | Type D | Type K"
            ],
            "confidence": "0.0 to 1.0",
            "reasoning_summary": "brief evidence-based explanation",
            "data_gaps": ["missing or uncertain information"],
        },
    }

    return (
        "# Gemini Incident Context\n\n"
        "Use this incident context to assess severity, likely incident type, "
        "what may be burning or affected, escalation risk, and recommended response. "
        "Weather conditions are provided when available: wind speed and the "
        "fire_spread_direction field indicate where the fire is most likely to travel. "
        "Treat the nearby-place data as evidence, not certainty.\n\n"
        "```json\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact Markdown context file for Gemini from nearby-place CSV data."
    )
    parser.add_argument("--incident-id", help="Incident ID to prepare. Defaults to first valid incident.")
    parser.add_argument("--nearby-csv", default=DEFAULT_NEARBY_CSV)
    parser.add_argument("--incidents-csv", default=DEFAULT_INCIDENTS_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-places", type=int, default=40)
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Do not fetch live weather conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    incident = load_incident_row(args.incidents_csv, args.incident_id)
    incident_id = str(incident["incidentId"])

    nearby = pd.read_csv(args.nearby_csv)
    incident_nearby = nearby[nearby["incident_id"].astype(str) == incident_id].copy()

    incident_nearby["distance_to_fire_m"] = pd.to_numeric(
        incident_nearby["distance_to_fire_m"],
        errors="coerce",
    )
    incident_nearby = incident_nearby.sort_values("distance_to_fire_m")

    weather = None
    if not args.skip_weather:
        from weather_enrichment import get_weather

        lat, lon = get_incident_coords(incident)
        if lat is not None and lon is not None:
            weather = get_weather(lat, lon)

    output = Path(args.output)
    output.write_text(
        build_markdown(incident, incident_nearby, args.max_places, weather),
        encoding="utf-8",
    )
    print(f"Saved Gemini context for {incident_id} to {output}")


if __name__ == "__main__":
    main()