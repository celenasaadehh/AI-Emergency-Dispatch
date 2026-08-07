import os
import subprocess
import sys
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

# west,south,east,north
LEBANON_BBOX = "35.0,33.0,36.7,34.7"
CALIFORNIA_BBOX = "-125,32,-114,42"  # for testing, fires are common here

SATELLITE = "VIIRS_NOAA20_NRT"


def fetch_fires(bbox, day_range=1):
    """Pull active fire detections from NASA FIRMS for a bounding box."""
    key = os.getenv("NASA_FIRMS_KEY")
    if not key:
        raise RuntimeError("NASA_FIRMS_KEY not set in .env.local")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{key}/{SATELLITE}/{bbox}/{day_range}"
    )
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        print(f"  FIRMS error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        return df

    # keep only nominal and high confidence detections
    if "confidence" in df.columns:
        df = df[df["confidence"].isin(["n", "h"])]

    return df


def severity_from_brightness(brightness):
    """Map thermal brightness (Kelvin) to a 0-100 severity score."""
    try:
        b = float(brightness)
    except (TypeError, ValueError):
        return 50
    return int(min(100, max(10, (b - 280) * 1.2)))


def to_incidents_csv(df, output_path="satellite_incidents.csv"):
    """Convert FIRMS detections into the incident CSV format the pipeline expects."""
    if df.empty:
        print("No satellite fires detected. Nothing to dispatch.")
        return None

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "incidentId": f"SAT{datetime.now().strftime('%Y%m%d')}{i:03d}",
            "eventType": "FIRE",
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "severity": severity_from_brightness(r.get("bright_ti4")),
            "radius_m": 500,
            "messageText": (
                f"Satellite detection ({SATELLITE}) on {r.get('acq_date')} "
                f"at {str(r.get('acq_time')).zfill(4)} UTC. "
                f"Confidence: {r.get('confidence')}. "
                f"Brightness: {r.get('bright_ti4')}K. No human report."
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} satellite incident(s) to {output_path}")
    return output_path


def main():
    # use --test to scan California instead of Lebanon
    testing = "--test" in sys.argv
    bbox = CALIFORNIA_BBOX if testing else LEBANON_BBOX
    region = "California (TEST)" if testing else "Lebanon"

    print(f"Scanning {region} for satellite-detected fires...")
    fires = fetch_fires(bbox, day_range=2)
    print(f"FIRMS returned {len(fires)} detection(s)")

    csv_path = to_incidents_csv(fires)
    if not csv_path:
        return

    # take the most severe fire and run it through the pipeline
    df = pd.read_csv(csv_path)
    worst = df.sort_values("severity", ascending=False).iloc[0]
    print(f"\nDispatching most severe incident: {worst['incidentId']} "
          f"(severity {worst['severity']})")

    subprocess.run([
        sys.executable, "run_incident_pipeline.py",
        "--input", csv_path,
        "--incident-id", str(worst["incidentId"]),
        "--places-source", "osm",
        "--skip-llm",
    ])


if __name__ == "__main__":
    main()