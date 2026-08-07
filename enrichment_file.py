import argparse
import json
import os
import time
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import requests


ENV_FILE = Path(__file__).with_name(".env.local")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
GOOGLE_PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
HEADERS = {
    "User-Agent": "fire-project/1.0"
}
GOOGLE_MAPS_API_KEY_ENV = "GOOGLE_MAPS_API_KEY"
GOOGLE_PLACES_FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.location,"
    "places.formattedAddress,"
    "places.primaryType,"
    "places.types"
)

MAX_RADIUS = 300
TOP_K = 3
DEFAULT_NEARBY_RADIUS_M = 200
MAX_NEARBY_RADIUS_M = 1000
OVERPASS_PAUSE_SECONDS = 1.0
OBJECTS_OUTPUT_COLUMNS = [
    "incident_id",
    "source",
    "place_name",
    "distance_to_fire_m",
    "place_type",
    "place_role",
    "matched_within_radius_m",
    "address",
    "additional_info",
    "osm_type",
    "osm_id",
    "google_place_id",
    "place_latitude",
    "place_longitude",
]

HAZARDS = {
    "fuel": '["amenity"="fuel"]',
    "restaurant": '["amenity"="restaurant"]',
    "power": '["power"~"substation|transformer|generator"]',
    "vegetation": '["natural"="wood"]'
}

# OSM tags are open-ended, so do not pre-filter by a fixed list such as
# building/shop/amenity. Fetch every tagged object in R, then classify later.
ANY_TAGGED_OSM_OBJECT_FILTER = '[~"."~"."]'
DEFAULT_EXCLUDED_TAG_KEYS = {"highway", "railway", "boundary", "route"}
DEFAULT_EXCLUDED_TAG_VALUES = {
    ("natural", "tree"),
    ("natural", "tree_row"),
    ("natural", "coastline"),
    ("natural", "wood"),
    ("natural", "scrub"),
    ("natural", "grassland"),
    ("place", "sea"),
    ("type", "boundary"),
    ("type", "route"),
    ("route", "bus"),
    ("route", "road"),
    ("route", "ferry"),
    ("landuse", "grass"),
    ("landuse", "forest"),
    ("landuse", "meadow"),
    ("landuse", "residential"),
    ("leisure", "garden"),
    ("leisure", "park"),
    ("barrier", "wall"),
    ("barrier", "fence"),
    ("barrier", "kerb"),
    ("amenity", "bench"),
    ("man_made", "street_lamp"),
    ("man_made", "utility_pole"),
}


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


load_local_env()

COORDINATE_COLUMN_CANDIDATES = [
    ("latitude", "longitude"),
    ("lat", "lon"),
    ("lat", "lng"),
    ("report_latitude", "report_longitude"),
    ("seed_lat", "seed_lon"),
]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius_m * c


def build_query(lat: float, lon: float, radius: int, tag: str) -> str:
    return f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon}){tag};
      way(around:{radius},{lat},{lon}){tag};
    );
    out center;
    """


def build_nearby_objects_query(lat: float, lon: float, radius: int) -> str:
    return f"""
    [out:json][timeout:30];
    (
      nwr(around:{radius},{lat},{lon}){ANY_TAGGED_OSM_OBJECT_FILTER};
    );
    out center;
    """


def send_query(query: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=HEADERS,
                timeout=40
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 502, 503, 504):
                wait_time = 8 * (attempt + 1)
                print(f"Server busy ({status}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

        except requests.exceptions.RequestException:
            wait_time = 8 * (attempt + 1)
            print(f"Request failed. Waiting {wait_time}s...")
            time.sleep(wait_time)

    return {"elements": []}


def extract_element_coordinates(element: dict) -> Tuple[Optional[float], Optional[float]]:
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    if (
        "center" in element
        and "lat" in element["center"]
        and "lon" in element["center"]
    ):
        return element["center"]["lat"], element["center"]["lon"]

    return None, None


def detect_coordinate_columns(
    df: pd.DataFrame,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> Tuple[str, str]:
    if lat_col and lon_col:
        missing_cols = {lat_col, lon_col} - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required coordinate columns: {missing_cols}")
        return lat_col, lon_col

    for candidate_lat, candidate_lon in COORDINATE_COLUMN_CANDIDATES:
        if candidate_lat in df.columns and candidate_lon in df.columns:
            return candidate_lat, candidate_lon

    raise ValueError(
        "Could not detect coordinate columns. Pass --lat-col and --lon-col explicitly."
    )


def safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_radius(radius_m: float) -> int:
    return int(max(1, min(MAX_NEARBY_RADIUS_M, round(radius_m))))


def radius_from_severity(severity: Any) -> int:
    if severity is None or pd.isna(severity):
        return DEFAULT_NEARBY_RADIUS_M

    if isinstance(severity, str):
        normalized = severity.strip().lower()
        severity_labels = {
            "low": 100,
            "minor": 100,
            "medium": 250,
            "moderate": 250,
            "high": 500,
            "severe": 500,
            "critical": 1000,
            "extreme": 1000,
        }
        if normalized in severity_labels:
            return severity_labels[normalized]
        numeric_value = safe_float(normalized)
    else:
        numeric_value = safe_float(severity)

    if numeric_value is None:
        return DEFAULT_NEARBY_RADIUS_M

    if 0 <= numeric_value <= 1:
        numeric_value *= 100

    if numeric_value < 40:
        return 100
    if numeric_value < 65:
        return 250
    if numeric_value < 85:
        return 500
    return 1000


def radius_for_row(
    row: pd.Series,
    default_radius_m: int,
    radius_col: Optional[str] = None,
    severity_col: Optional[str] = None,
) -> int:
    if radius_col and radius_col in row:
        radius_value = safe_float(row[radius_col])
        if radius_value is not None:
            return clamp_radius(radius_value)

    if severity_col and severity_col in row:
        return radius_from_severity(row[severity_col])

    return clamp_radius(default_radius_m)


def select_relevant_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    useful_prefixes = ("addr:", "contact:", "name:")
    useful_keys = {
        "name",
        "brand",
        "operator",
        "building",
        "shop",
        "amenity",
        "office",
        "craft",
        "industrial",
        "tourism",
        "leisure",
        "healthcare",
        "emergency",
        "military",
        "man_made",
        "power",
        "landuse",
        "natural",
        "opening_hours",
        "phone",
        "website",
        "description",
        "highway",
        "railway",
        "boundary",
        "route",
        "type",
        "barrier",
        "place",
        "content",
        "substance",
        "hazard",
        "hazmat",
        "material",
        "product",
        "goods",
        "google_primary_type",
        "google_types",
    }
    return {
        key: value
        for key, value in tags.items()
        if key in useful_keys or key.startswith(useful_prefixes)
    }


def object_role(tags: Dict[str, Any]) -> str:
    if tags.get("building"):
        return "building"
    if tags.get("shop") or tags.get("craft"):
        return "business"
    if (
        tags.get("amenity")
        or tags.get("office")
        or tags.get("healthcare")
        or tags.get("tourism")
        or tags.get("leisure")
    ):
        return "place"
    if tags.get("emergency"):
        return "emergency"
    if tags.get("industrial") or tags.get("landuse") == "industrial":
        return "industrial"
    if tags.get("military"):
        return "military"
    if tags.get("man_made"):
        return "infrastructure"
    if tags.get("power"):
        return "power"
    if tags.get("highway"):
        return "road"
    if tags.get("railway"):
        return "railway"
    if tags.get("natural"):
        return "natural"
    return "mapped_object"


def classify_osm_object(tags: Dict[str, Any]) -> str:
    if tags.get("shop") == "mall":
        return "mall"
    if tags.get("shop"):
        return f"shop:{tags['shop']}"
    if tags.get("amenity"):
        return f"amenity:{tags['amenity']}"
    if tags.get("office"):
        return f"office:{tags['office']}"
    if tags.get("healthcare"):
        return f"healthcare:{tags['healthcare']}"
    if tags.get("emergency"):
        return f"emergency:{tags['emergency']}"
    if tags.get("tourism"):
        return f"tourism:{tags['tourism']}"
    if tags.get("leisure"):
        return f"leisure:{tags['leisure']}"
    if tags.get("craft"):
        return f"craft:{tags['craft']}"
    if tags.get("industrial"):
        return f"industrial:{tags['industrial']}"
    if tags.get("building"):
        return f"building:{tags['building']}"
    if tags.get("military"):
        return f"military:{tags['military']}"
    if tags.get("man_made"):
        return f"man_made:{tags['man_made']}"
    if tags.get("landuse"):
        return f"landuse:{tags['landuse']}"
    if tags.get("power"):
        return f"power:{tags['power']}"
    if tags.get("highway"):
        return f"highway:{tags['highway']}"
    if tags.get("railway"):
        return f"railway:{tags['railway']}"
    if tags.get("natural"):
        return f"natural:{tags['natural']}"
    return "other"


def display_name_for_object(tags: Dict[str, Any], category: str) -> str:
    for key in ("name:en", "name", "brand", "operator", "addr:housename"):
        value = tags.get(key)
        if value:
            return str(value)

    if category.startswith("building:"):
        return "Unnamed building"
    return f"Unnamed {category}"


def address_from_tags(tags: Dict[str, Any]) -> Optional[str]:
    parts = []
    street = tags.get("addr:street")
    house_number = tags.get("addr:housenumber")
    city = tags.get("addr:city")

    if house_number and street:
        parts.append(f"{house_number} {street}")
    elif street:
        parts.append(str(street))
    elif house_number:
        parts.append(str(house_number))

    if city:
        parts.append(str(city))

    return ", ".join(parts) if parts else None


def format_nearby_summary(objects: List[Dict[str, Any]], max_items: int = 12) -> str:
    summaries = []
    for obj in objects[:max_items]:
        source = obj.get("source", "osm")
        if source == "google":
            source_ref = obj.get("google_place_id", "")
        else:
            source_ref = f"{obj.get('osm_type')}/{obj.get('osm_id')}"
        summaries.append(
            f"{obj['distance_m']}m {obj['role']} {obj['category']} {obj['name']} "
            f"({source}:{source_ref})"
        )

    if len(objects) > max_items:
        summaries.append(f"... {len(objects) - max_items} more")

    return "; ".join(summaries)


def is_default_noise_object(obj: Dict[str, Any]) -> bool:
    tags = obj.get("tags", {})
    if not tags:
        return True

    if obj.get("category") == "other" and str(obj.get("name", "")).startswith("Unnamed"):
        return True

    if any(key in tags for key in DEFAULT_EXCLUDED_TAG_KEYS):
        return True

    for key, value in DEFAULT_EXCLUDED_TAG_VALUES:
        if tags.get(key) == value:
            return True

    if obj.get("osm_type") == "relation" and not has_operational_signal(tags):
        return True

    distance_m = safe_float(obj.get("distance_m"))
    matched_radius_m = safe_float(obj.get("matched_within_radius_m"))
    if (
        distance_m is not None
        and matched_radius_m is not None
        and distance_m > matched_radius_m * 3
        and not has_operational_signal(tags)
    ):
        return True

    return False


def filter_nearby_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [obj for obj in objects if not is_default_noise_object(obj)]


def has_operational_signal(tags: Dict[str, Any]) -> bool:
    useful_keys = {
        "building",
        "shop",
        "amenity",
        "office",
        "craft",
        "industrial",
        "tourism",
        "leisure",
        "healthcare",
        "emergency",
        "military",
        "man_made",
        "power",
    }
    return any(key in tags for key in useful_keys)


def compact_tags(tags: Dict[str, Any]) -> str:
    preferred_keys = [
        "building",
        "shop",
        "amenity",
        "office",
        "craft",
        "industrial",
        "tourism",
        "leisure",
        "healthcare",
        "emergency",
        "military",
        "man_made",
        "power",
        "operator",
        "brand",
        "opening_hours",
        "phone",
        "contact:phone",
        "website",
        "contact:website",
        "content",
        "substance",
        "hazard",
        "hazmat",
        "material",
        "product",
        "goods",
        "google_primary_type",
        "google_types",
    ]
    parts = []
    for key in preferred_keys:
        value = tags.get(key)
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def get_distances_to_feature(
    lat: float,
    lon: float,
    tag: str,
    max_radius: int = MAX_RADIUS
) -> List[float]:
    query = build_query(lat, lon, max_radius, tag)
    data = send_query(query)

    distances: List[float] = []

    if "elements" not in data:
        return distances

    for element in data["elements"]:
        elem_lat, elem_lon = extract_element_coordinates(element)

        if elem_lat is None or elem_lon is None:
            continue

        distance = haversine_m(lat, lon, elem_lat, elem_lon)
        if distance <= max_radius:
            distances.append(distance)

    distances.sort()
    return distances


def fetch_nearby_objects(
    lat: float,
    lon: float,
    radius_m: int = DEFAULT_NEARBY_RADIUS_M,
    max_results: Optional[int] = None,
    include_full_tags: bool = False,
) -> List[Dict[str, Any]]:
    radius_m = clamp_radius(radius_m)
    query = build_nearby_objects_query(lat, lon, radius_m)
    data = send_query(query)

    objects: List[Dict[str, Any]] = []
    seen = set()

    for element in data.get("elements", []):
        osm_type = element.get("type")
        osm_id = element.get("id")
        if osm_type is None or osm_id is None:
            continue

        object_key = (osm_type, osm_id)
        if object_key in seen:
            continue
        seen.add(object_key)

        elem_lat, elem_lon = extract_element_coordinates(element)
        if elem_lat is None or elem_lon is None:
            continue

        center_distance = haversine_m(lat, lon, elem_lat, elem_lon)
        if center_distance > radius_m:
            continue

        tags = element.get("tags", {})
        relevant_tags = dict(tags) if include_full_tags else select_relevant_tags(tags)
        category = classify_osm_object(relevant_tags)
        role = object_role(relevant_tags)

        objects.append(
            {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "google_place_id": None,
                "source": "osm",
                "name": display_name_for_object(relevant_tags, category),
                "role": role,
                "category": category,
                "distance_m": round(center_distance, 2),
                "matched_within_radius_m": radius_m,
                "latitude": round(float(elem_lat), 7),
                "longitude": round(float(elem_lon), 7),
                "address": address_from_tags(relevant_tags),
                "tags": relevant_tags,
            }
        )

    objects.sort(key=lambda obj: obj["distance_m"])
    if max_results is not None:
        return objects[:max_results]

    return objects


def fetch_google_nearby_places(
    lat: float,
    lon: float,
    radius_m: int = DEFAULT_NEARBY_RADIUS_M,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    api_key = os.getenv(GOOGLE_MAPS_API_KEY_ENV)
    if not api_key:
        print(f"{GOOGLE_MAPS_API_KEY_ENV} not set; skipping Google Places")
        return []

    radius_m = clamp_radius(radius_m)
    result_limit = min(max_results or 20, 20)
    payload = {
        "maxResultCount": result_limit,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lon,
                },
                "radius": float(radius_m),
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": GOOGLE_PLACES_FIELD_MASK,
    }

    try:
        response = requests.post(
            GOOGLE_PLACES_NEARBY_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        print(f"Google Places request failed ({status}); skipping Google Places")
        return []
    except requests.exceptions.RequestException:
        print("Google Places request failed; skipping Google Places")
        return []

    places = []
    for place in response.json().get("places", []):
        location = place.get("location", {})
        place_lat = safe_float(location.get("latitude"))
        place_lon = safe_float(location.get("longitude"))
        if place_lat is None or place_lon is None:
            continue

        distance_m = haversine_m(lat, lon, place_lat, place_lon)
        if distance_m > radius_m:
            continue

        display_name = place.get("displayName") or {}
        name = display_name.get("text") or "Unnamed Google place"
        primary_type = place.get("primaryType") or "google_place"
        place_types = place.get("types") or []
        tags = {
            "google_primary_type": primary_type,
            "google_types": "|".join(place_types),
        }

        places.append(
            {
                "osm_type": None,
                "osm_id": None,
                "google_place_id": place.get("id"),
                "source": "google",
                "name": name,
                "role": "place",
                "category": primary_type,
                "distance_m": round(distance_m, 2),
                "matched_within_radius_m": radius_m,
                "latitude": round(float(place_lat), 7),
                "longitude": round(float(place_lon), 7),
                "address": place.get("formattedAddress"),
                "tags": tags,
            }
        )

    places.sort(key=lambda obj: obj["distance_m"])
    return places


def dedupe_nearby_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_google_ids = set()
    seen_osm_ids = set()
    seen_approx_places = set()
    deduped = []

    for obj in sorted(objects, key=lambda item: item["distance_m"]):
        source = obj.get("source")
        google_place_id = obj.get("google_place_id")
        osm_key = (obj.get("osm_type"), obj.get("osm_id"))

        if source == "google" and google_place_id:
            if google_place_id in seen_google_ids:
                continue
            seen_google_ids.add(google_place_id)
        elif all(osm_key):
            if osm_key in seen_osm_ids:
                continue
            seen_osm_ids.add(osm_key)

        approx_key = (
            str(obj.get("name", "")).strip().lower(),
            round(float(obj.get("latitude")), 4),
            round(float(obj.get("longitude")), 4),
        )
        if approx_key in seen_approx_places:
            continue
        seen_approx_places.add(approx_key)
        deduped.append(obj)

    return deduped


def nearby_objects_features(
    lat: float,
    lon: float,
    radius_m: int = DEFAULT_NEARBY_RADIUS_M,
    max_results: Optional[int] = None,
    apply_default_filter: bool = True,
    include_full_tags: bool = False,
    places_source: str = "both",
) -> Dict[str, Any]:
    osm_objects: List[Dict[str, Any]] = []
    google_objects: List[Dict[str, Any]] = []
    if places_source in ("osm", "both"):
        osm_objects = fetch_nearby_objects(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            include_full_tags=include_full_tags,
        )
    if places_source in ("google", "both"):
        google_objects = fetch_google_nearby_places(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            max_results=max_results,
        )

    raw_objects = osm_objects + google_objects
    filtered_osm_objects = (
        filter_nearby_objects(osm_objects) if apply_default_filter else osm_objects
    )
    filtered_objects = dedupe_nearby_objects(filtered_osm_objects + google_objects)
    objects = filtered_objects
    if max_results is not None:
        objects = filtered_objects[:max_results]

    building_count = sum(1 for obj in objects if obj["role"] == "building")
    business_count = sum(1 for obj in objects if obj["role"] == "business")
    place_count = sum(1 for obj in objects if obj["role"] == "place")
    role_counts: Dict[str, int] = {}
    for obj in objects:
        role = obj["role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "nearby_radius_m": radius_m,
        "nearby_objects_raw_count": len(raw_objects),
        "nearby_osm_objects_count": len(osm_objects),
        "nearby_google_places_count": len(google_objects),
        "nearby_objects_filtered_out_count": len(raw_objects) - len(filtered_objects),
        "nearby_objects_limited_out_count": len(filtered_objects) - len(objects),
        "nearby_objects_count": len(objects),
        "nearby_buildings_count": building_count,
        "nearby_businesses_count": business_count,
        "nearby_places_count": place_count,
        "nearby_roles_json": json.dumps(role_counts, ensure_ascii=False),
        "nearby_objects_json": json.dumps(objects, ensure_ascii=False),
        "nearby_objects_summary": format_nearby_summary(objects),
    }


def source_id_from_row(row: pd.Series, source_row_number: int) -> str:
    for col in ("incidentId", "incident_id", "report_id", "id"):
        if col in row and not pd.isna(row[col]):
            return str(row[col])
    return f"row_{source_row_number}"


def append_object_rows(
    object_rows: List[Dict[str, Any]],
    source_row: pd.Series,
    source_row_number: int,
    objects_json: str,
) -> None:
    source_id = source_id_from_row(source_row, source_row_number)
    objects = json.loads(objects_json)

    for obj in objects:
        object_rows.append(
            {
                "incident_id": source_id,
                "source": obj.get("source", "osm"),
                "place_name": obj.get("name"),
                "distance_to_fire_m": obj.get("distance_m"),
                "place_type": obj.get("category"),
                "place_role": obj.get("role"),
                "matched_within_radius_m": obj.get("matched_within_radius_m"),
                "address": obj.get("address"),
                "additional_info": compact_tags(obj.get("tags", {})),
                "osm_type": obj.get("osm_type"),
                "osm_id": obj.get("osm_id"),
                "google_place_id": obj.get("google_place_id"),
                "place_latitude": obj.get("latitude"),
                "place_longitude": obj.get("longitude"),
            }
        )


def build_distance_features(
    distances: List[float],
    prefix: str,
    top_k: int = TOP_K,
    max_radius: int = MAX_RADIUS
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for i in range(top_k):
        col_name = f"{prefix}_dist_{i + 1}"
        if i < len(distances):
            features[col_name] = round(distances[i], 2)
        else:
            features[col_name] = -1.0

    features[f"{prefix}_count_{max_radius}m"] = len(distances)
    return features


def enrich_location(lat: float, lon: float) -> Dict[str, float]:
    all_features: Dict[str, float] = {}

    for hazard_name, tag in HAZARDS.items():
        distances = get_distances_to_feature(lat, lon, tag, max_radius=MAX_RADIUS)
        hazard_features = build_distance_features(
            distances=distances,
            prefix=hazard_name,
            top_k=TOP_K,
            max_radius=MAX_RADIUS
        )
        all_features.update(hazard_features)

        # polite pause for free API
        time.sleep(OVERPASS_PAUSE_SECONDS)

    return all_features


def enrich_location_with_nearby_objects(
    lat: float,
    lon: float,
    radius_m: int = DEFAULT_NEARBY_RADIUS_M,
    max_results: Optional[int] = None,
    apply_default_filter: bool = True,
    include_full_tags: bool = False,
    places_source: str = "both",
) -> Dict[str, Any]:
    return nearby_objects_features(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        max_results=max_results,
        apply_default_filter=apply_default_filter,
        include_full_tags=include_full_tags,
        places_source=places_source,
    )


def enrich_dataset(
    input_csv: str,
    output_csv: str,
    lat_col: str = "report_latitude",
    lon_col: str = "report_longitude"
) -> None:
    df = pd.read_csv(input_csv)

    required_cols = {lat_col, lon_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    enriched_rows: List[Dict[str, float]] = []

    for i, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        print(f"Processing row {i + 1}/{len(df)} at ({lat}, {lon})")
        features = enrich_location(lat, lon)
        enriched_rows.append(features)

    features_df = pd.DataFrame(enriched_rows)
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    final_df.to_csv(output_csv, index=False)
    print(f"Done. Saved enriched dataset to: {output_csv}")


def enrich_dataset_with_nearby_objects(
    input_csv: str,
    output_csv: str,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    radius_m: int = DEFAULT_NEARBY_RADIUS_M,
    radius_col: Optional[str] = None,
    severity_col: Optional[str] = None,
    max_results: Optional[int] = None,
    apply_default_filter: bool = True,
    include_full_tags: bool = False,
    places_source: str = "both",
    objects_output_csv: Optional[str] = None,
    summary_output_csv: Optional[str] = None,
) -> None:
    df = pd.read_csv(input_csv)
    detected_lat_col, detected_lon_col = detect_coordinate_columns(df, lat_col, lon_col)

    if radius_col and radius_col not in df.columns:
        raise ValueError(f"Missing radius column: {radius_col}")
    if severity_col and severity_col not in df.columns:
        raise ValueError(f"Missing severity column: {severity_col}")

    enriched_rows: List[Dict[str, Any]] = []
    object_rows: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        lat = safe_float(row[detected_lat_col])
        lon = safe_float(row[detected_lon_col])

        if lat is None or lon is None:
            print(f"Skipping row {i + 1}/{len(df)} because coordinates are invalid")
            enriched_rows.append(
                {
                    "nearby_radius_m": None,
                    "nearby_objects_raw_count": 0,
                    "nearby_osm_objects_count": 0,
                    "nearby_google_places_count": 0,
                    "nearby_objects_filtered_out_count": 0,
                    "nearby_objects_limited_out_count": 0,
                    "nearby_objects_count": 0,
                    "nearby_buildings_count": 0,
                    "nearby_businesses_count": 0,
                    "nearby_places_count": 0,
                    "nearby_roles_json": "{}",
                    "nearby_objects_json": "[]",
                    "nearby_objects_summary": "",
                }
            )
            continue

        row_radius_m = radius_for_row(
            row=row,
            default_radius_m=radius_m,
            radius_col=radius_col,
            severity_col=severity_col,
        )

        print(
            f"Processing row {i + 1}/{len(df)} at ({lat}, {lon}) "
            f"with radius {row_radius_m}m"
        )
        features = enrich_location_with_nearby_objects(
            lat=lat,
            lon=lon,
            radius_m=row_radius_m,
            max_results=max_results,
            apply_default_filter=apply_default_filter,
            include_full_tags=include_full_tags,
            places_source=places_source,
        )
        enriched_rows.append(features)
        append_object_rows(
            object_rows=object_rows,
            source_row=row,
            source_row_number=i + 1,
            objects_json=features["nearby_objects_json"],
        )

        # Public Overpass is shared infrastructure; keep batch processing polite.
        time.sleep(OVERPASS_PAUSE_SECONDS)

    features_df = pd.DataFrame(enriched_rows)
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    readable_output_csv = objects_output_csv or output_csv
    objects_df = pd.DataFrame(object_rows, columns=OBJECTS_OUTPUT_COLUMNS)
    if not objects_df.empty:
        objects_df = objects_df.sort_values(
            by=["incident_id", "distance_to_fire_m", "source", "place_name"],
            na_position="last",
        )
    objects_df.to_csv(readable_output_csv, index=False)
    print(f"Done. Saved readable nearby places table to: {readable_output_csv}")

    if summary_output_csv:
        final_df.to_csv(summary_output_csv, index=False)
        print(f"Done. Saved nearby-object summary dataset to: {summary_output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich incident/report CSV rows with nearby OpenStreetMap objects "
            "and/or Google Places inside a radius."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output CSV path. In nearby mode this is the readable one-place-per-row "
            "table; in hazards mode it is the older feature table."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("nearby", "hazards"),
        default="nearby",
        help=(
            "nearby lists tagged OSM objects and/or Google Places inside the radius; "
            "hazards keeps the older distance features"
        ),
    )
    parser.add_argument("--lat-col", help="Latitude column name")
    parser.add_argument("--lon-col", help="Longitude column name")
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_NEARBY_RADIUS_M,
        help="Default search radius in meters for nearby mode",
    )
    parser.add_argument(
        "--places-source",
        choices=("osm", "google", "both"),
        default="both",
        help="Nearby data source. Default combines OSM and Google Places.",
    )
    parser.add_argument(
        "--radius-col",
        help="Optional per-row radius column in meters; overrides --radius",
    )
    parser.add_argument(
        "--severity-col",
        help="Optional severity column; used to derive radius if --radius-col is absent",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Optional cap on objects stored per row; omit to keep all results",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help=(
            "Keep road/railway/tree objects in the output. By default they are "
            "removed after fetching to keep the CSV smaller."
        ),
    )
    parser.add_argument(
        "--full-tags",
        action="store_true",
        help="Store every OSM tag in output JSON instead of a compact useful subset.",
    )
    parser.add_argument(
        "--objects-output",
        help=(
            "Optional readable long-format CSV path. If omitted, nearby mode writes "
            "this readable table to --output."
        ),
    )
    parser.add_argument(
        "--summary-output",
        help=(
            "Optional wide CSV path with the original input rows plus nearby-object "
            "counts, JSON, and summaries."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "hazards":
        enrich_dataset(
            input_csv=args.input,
            output_csv=args.output,
            lat_col=args.lat_col or "report_latitude",
            lon_col=args.lon_col or "report_longitude",
        )
    else:
        enrich_dataset_with_nearby_objects(
            input_csv=args.input,
            output_csv=args.output,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            radius_m=args.radius,
            radius_col=args.radius_col,
            severity_col=args.severity_col,
            max_results=args.max_results,
            apply_default_filter=not args.include_noise,
            include_full_tags=args.full_tags,
            places_source=args.places_source,
            objects_output_csv=args.objects_output,
            summary_output_csv=args.summary_output,
        )
