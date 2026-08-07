import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

from dispatch_recommendations import (
    DEFAULT_COMPACT_OUTPUT_JSON,
    DEFAULT_FIRE_STATIONS_JSON,
    DEFAULT_FULL_OUTPUT_JSON,
    write_dispatch_outputs,
)
from enrichment_file import (
    DEFAULT_NEARBY_RADIUS_M,
    GOOGLE_MAPS_API_KEY_ENV,
    enrich_dataset_with_nearby_objects,
    load_local_env,
)
from generate_gemini_report import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    GEMINI_API_KEY_ENV,
    call_gemini_with_fallbacks,
)
from prepare_gemini_context import build_markdown, get_incident_coords, load_incident_row
from weather_enrichment import get_weather


DEFAULT_OUTPUT_DIR = "pipeline_output"


def validate_input_csv(input_csv: str, incident_id: Optional[str]) -> str:
    df = pd.read_csv(input_csv)
    if "incidentId" not in df.columns:
        if len(df) != 1:
            raise ValueError(
                "Input CSV needs an incidentId column, or it must contain exactly one row."
            )
        df["incidentId"] = "INC001"
        df.to_csv(input_csv, index=False)

    if incident_id:
        matches = df[df["incidentId"].astype(str) == str(incident_id)]
        if matches.empty:
            raise ValueError(f"Incident {incident_id} was not found in {input_csv}")
        return str(incident_id)

    first_valid = df[df["incidentId"].notna()]
    if first_valid.empty:
        raise ValueError("Input CSV does not contain a valid incidentId")
    return str(first_valid.iloc[0]["incidentId"])


def ensure_secret(name: str, placeholder: str) -> str:
    value = os.getenv(name)
    if not value or value == placeholder:
        raise RuntimeError(f"Set {name} in .env.local before running the pipeline.")
    return value


def prepare_context_file(
    input_csv: str,
    nearby_csv: str,
    incident_id: str,
    output_md: Path,
    max_places: int,
    weather: Optional[dict] = None,
) -> None:
    incident = load_incident_row(input_csv, incident_id)
    nearby = pd.read_csv(nearby_csv)
    incident_nearby = nearby[nearby["incident_id"].astype(str) == incident_id].copy()

    incident_nearby["distance_to_fire_m"] = pd.to_numeric(
        incident_nearby["distance_to_fire_m"],
        errors="coerce",
    )
    incident_nearby = incident_nearby.sort_values("distance_to_fire_m")
    output_md.write_text(
        build_markdown(incident, incident_nearby, max_places, weather),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full incident workflow: input CSV -> OSM/Google nearby places "
            "-> live weather -> Gemini context -> Markdown emergency report."
        )
    )
    parser.add_argument("--input", required=True, help="User incident CSV path")
    parser.add_argument("--incident-id", help="Incident ID to process. Defaults to first valid row.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lat-col", help="Latitude column name if auto-detection fails")
    parser.add_argument("--lon-col", help="Longitude column name if auto-detection fails")
    parser.add_argument("--radius", type=int, default=DEFAULT_NEARBY_RADIUS_M)
    parser.add_argument("--radius-col", help="Optional per-row radius column in meters")
    parser.add_argument("--severity-col", help="Optional severity column used to derive radius")
    parser.add_argument(
        "--places-source",
        choices=("osm", "google", "both"),
        default="both",
        help="Nearby data source. Default combines OSM and Google Places.",
    )
    parser.add_argument("--max-places", type=int, default=40)
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Do not fetch live weather conditions for the incident location.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--fallback-model",
        action="append",
        default=DEFAULT_FALLBACK_MODELS,
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--fire-stations",
        default=DEFAULT_FIRE_STATIONS_JSON,
        help="Fire station equipment database JSON path.",
    )
    parser.add_argument(
        "--dispatch-output",
        help=(
            "Full dispatch recommendation JSON path. "
            "Defaults to dispatch_recommendations.json inside --output-dir."
        ),
    )
    parser.add_argument(
        "--dispatch-compact-output",
        help=(
            "Compact front-end recommendation JSON path. "
            "Defaults to dispatch_recommendations_frontend.json inside --output-dir."
        ),
    )
    parser.add_argument(
        "--required-equipment",
        action="append",
        help=(
            "Override required equipment for dispatch JSON. Can be repeated or "
            "comma-separated, for example: --required-equipment 'Type A,Type C'."
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Generate nearby CSV, Gemini context, and dispatch JSON only; do not call Gemini.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()

    input_csv = Path(args.input)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    incident_id = validate_input_csv(str(input_csv), args.incident_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_input_csv = output_dir / "incident_input.csv"
    nearby_csv = output_dir / "incident_nearby_places_structured.csv"
    context_md = output_dir / "gemini_incident_context.md"
    report_md = output_dir / "incident_report.md"
    dispatch_output = (
        Path(args.dispatch_output)
        if args.dispatch_output
        else output_dir / Path(DEFAULT_FULL_OUTPUT_JSON).name
    )
    dispatch_compact_output = (
        Path(args.dispatch_compact_output)
        if args.dispatch_compact_output
        else output_dir / Path(DEFAULT_COMPACT_OUTPUT_JSON).name
    )
    shutil.copyfile(input_csv, run_input_csv)

    if args.places_source in ("google", "both"):
        ensure_secret(GOOGLE_MAPS_API_KEY_ENV, "put_your_google_maps_api_key_here")

    print("Step 1/6: generating OSM/Google nearby-place CSV")
    enrich_dataset_with_nearby_objects(
        input_csv=str(run_input_csv),
        output_csv=str(nearby_csv),
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        radius_m=args.radius,
        radius_col=args.radius_col,
        severity_col=args.severity_col,
        max_results=None,
        apply_default_filter=True,
        include_full_tags=False,
        places_source=args.places_source,
    )

    weather = None
    if args.skip_weather:
        print("Step 2/6: skipped weather fetch")
    else:
        print("Step 2/6: fetching live weather at incident location")
        incident_row = load_incident_row(str(run_input_csv), incident_id)
        lat, lon = get_incident_coords(incident_row)
        if lat is None or lon is None:
            print("  Warning: could not read incident coordinates, skipping weather")
        else:
            weather = get_weather(lat, lon)
            if weather:
                print(
                    f"  {weather['temperature_c']}C, humidity {weather['humidity_pct']}%, "
                    f"wind {weather['wind_speed_ms']} m/s -> fire likely spreads "
                    f"{weather['fire_spread_direction']}"
                )

    print("Step 3/6: preparing Gemini context Markdown")
    prepare_context_file(
        input_csv=str(run_input_csv),
        nearby_csv=str(nearby_csv),
        incident_id=incident_id,
        output_md=context_md,
        max_places=args.max_places,
        weather=weather,
    )
    print(f"Saved Gemini context to {context_md}")

    if args.skip_llm:
        print("Step 4/6: skipped Gemini call")
        print("Step 5/6: skipped Markdown report")
        print("Step 6/6: writing dispatch recommendation JSON")
        incident = load_incident_row(str(run_input_csv), incident_id)
        incident_dict = incident.to_dict()
        if args.required_equipment:
            incident_dict["__requiredEquipmentOverride"] = ",".join(args.required_equipment)
        write_dispatch_outputs(
            incident=incident_dict,
            stations_path=Path(args.fire_stations),
            full_output_path=dispatch_output,
            compact_output_path=dispatch_compact_output,
            context_text=context_md.read_text(encoding="utf-8"),
        )
        print(f"Done. Outputs are in {output_dir}")
        print(f"Saved dispatch recommendations to {dispatch_output}")
        print(f"Saved compact front-end dispatch JSON to {dispatch_compact_output}")
        return

    print("Step 4/6: calling Gemini")
    gemini_key = ensure_secret(GEMINI_API_KEY_ENV, "put_your_gemini_api_key_here")
    context = context_md.read_text(encoding="utf-8")
    models: List[str] = [args.model] + [
        fallback for fallback in args.fallback_model if fallback != args.model
    ]
    report = call_gemini_with_fallbacks(
        context=context,
        models=models,
        api_key=gemini_key,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    print("Step 5/6: writing report")
    report_md.write_text(report + "\n", encoding="utf-8")
    print("Step 6/6: writing dispatch recommendation JSON")
    incident = load_incident_row(str(run_input_csv), incident_id)
    incident_dict = incident.to_dict()
    if args.required_equipment:
        incident_dict["__requiredEquipmentOverride"] = ",".join(args.required_equipment)
    write_dispatch_outputs(
        incident=incident_dict,
        stations_path=Path(args.fire_stations),
        full_output_path=dispatch_output,
        compact_output_path=dispatch_compact_output,
        report_text=report,
        context_text=context,
    )
    print(f"Done. Saved report to {report_md}")
    print(f"Saved dispatch recommendations to {dispatch_output}")
    print(f"Saved compact front-end dispatch JSON to {dispatch_compact_output}")


if __name__ == "__main__":
    main()