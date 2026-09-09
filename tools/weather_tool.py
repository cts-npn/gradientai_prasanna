"""Live weather via Open-Meteo — free, no API key, no rate-limit registration.

Two real calls: geocode the city name to coordinates, then fetch current
conditions for those coordinates. Both responses are used as-is; nothing
here is invented.

API docs: https://open-meteo.com/en/docs / https://open-meteo.com/en/docs/geocoding-api
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.tool_utils import ToolRequestError, http_get, now_iso, ttl_cache

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_FIELDS = "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"

# WMO Weather interpretation codes, as documented by Open-Meteo.
_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass
class WeatherResult:
    resolved_name: str
    country: str | None
    latitude: float
    longitude: float
    temperature_c: float | None
    apparent_temperature_c: float | None
    humidity_pct: float | None
    wind_speed_kmh: float | None
    precipitation_mm: float | None
    condition: str
    observation_time: str | None  # ISO timestamp from Open-Meteo, local to the location
    source_url: str  # the actual forecast API call made, real and reproducible
    retrieved_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return {
            "resolved_name": self.resolved_name,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature_c": self.temperature_c,
            "apparent_temperature_c": self.apparent_temperature_c,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "precipitation_mm": self.precipitation_mm,
            "condition": self.condition,
            "observation_time": self.observation_time,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }


@ttl_cache
def _geocode(city: str) -> dict | None:
    """Resolve a city name to coordinates. Returns None if not found or on error."""
    try:
        data = http_get(GEOCODE_URL, params={"name": city, "count": 1, "language": "en", "format": "json"})
    except ToolRequestError:
        return None
    results = data.get("results") or []
    return results[0] if results else None


@ttl_cache
def get_weather(city: str) -> WeatherResult | None:
    """Fetch current weather for `city`. Returns None if the city can't be
    resolved or either API call fails — callers must treat that as "no
    weather evidence available," never substitute model knowledge.
    """
    if not city or not city.strip():
        return None

    place = _geocode(city.strip())
    if not place:
        return None

    lat, lon = place["latitude"], place["longitude"]
    params = {"latitude": lat, "longitude": lon, "current": CURRENT_FIELDS, "timezone": "auto"}
    try:
        forecast = http_get(FORECAST_URL, params=params)
    except ToolRequestError:
        return None

    current = forecast.get("current")
    if not current:
        return None

    code = current.get("weather_code")
    condition = _WEATHER_CODES.get(code, f"Unknown (WMO code {code})")

    import urllib.parse

    source_url = f"{FORECAST_URL}?{urllib.parse.urlencode(params)}"

    return WeatherResult(
        resolved_name=place.get("name", city),
        country=place.get("country"),
        latitude=lat,
        longitude=lon,
        temperature_c=current.get("temperature_2m"),
        apparent_temperature_c=current.get("apparent_temperature"),
        humidity_pct=current.get("relative_humidity_2m"),
        wind_speed_kmh=current.get("wind_speed_10m"),
        precipitation_mm=current.get("precipitation"),
        condition=condition,
        observation_time=current.get("time"),
        source_url=source_url,
    )
