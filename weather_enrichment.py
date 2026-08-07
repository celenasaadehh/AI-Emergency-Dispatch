import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _label(deg):
    if deg is None:
        return "unknown"
    return COMPASS[round(deg / 45) % 8]


def wind_from_label(deg):
    """Direction the wind is blowing FROM (what OpenWeather reports)."""
    return _label(deg)


def wind_toward_label(deg):
    """Direction the wind is blowing TOWARD -- this is where fire spreads."""
    if deg is None:
        return "unknown"
    return _label((deg + 180) % 360)


def get_weather(lat, lon):
    """Fetch current weather at the incident coordinates. Returns dict or None."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("  Warning: OPENWEATHER_API_KEY not set, skipping weather enrichment")
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        deg = data["wind"].get("deg")
        return {
            "temperature_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "wind_speed_ms": data["wind"]["speed"],
            "wind_gust_ms": data["wind"].get("gust"),
            "wind_from_deg": deg,
            "wind_from_direction": wind_from_label(deg),
            "fire_spread_direction": wind_toward_label(deg),
            "conditions": data["weather"][0]["description"],
            "visibility_m": data.get("visibility"),
        }
    except Exception as exc:
        print(f"  Warning: weather fetch failed: {exc}")
        return None


if __name__ == "__main__":
    w = get_weather(33.8938, 35.5018)
    if w:
        for k, v in w.items():
            print(f"  {k}: {v}")