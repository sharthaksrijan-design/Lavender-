"""
LAVENDER — Web Tools
tools/web_tools.py

Real-time information retrieval:
  - DuckDuckGo search (no API key required)
  - Web page fetch + clean text extraction
  - Weather via Open-Meteo (free, no key)
  - Basic news headlines

These are called by the LangGraph tool executor when the intent
router classifies a query as INFORMATIONAL_REALTIME.
"""

import json
import logging
import re
from typing import Optional
import httpx

logger = logging.getLogger("lavender.web_tools")

# DuckDuckGo instant answer API (no key, rate-limited but fine for personal use)
DDG_API_URL   = "https://api.duckduckgo.com/"
# Open-Meteo free weather API
WEATHER_URL   = "https://api.open-meteo.com/v1/forecast"
# Geocoding for weather (also free)
GEOCODE_URL   = "https://geocoding-api.open-meteo.com/v1/search"


class WebTools:
    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Lavender/1.0; personal AI)"
            },
            follow_redirects=True,
        )

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> str:
        """
        DuckDuckGo instant answer + abstract.
        Returns formatted string for the LLM to use.
        Falls back to a note that results were unavailable.
        """
        try:
            r = self._client.get(DDG_API_URL, params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            })
            data = r.json()

            parts = []

            # Instant answer (calculator, conversions, etc.)
            if data.get("Answer"):
                parts.append(f"Answer: {data['Answer']}")

            # Abstract from Wikipedia or other sources
            if data.get("AbstractText"):
                parts.append(f"Summary: {data['AbstractText']}")
                if data.get("AbstractSource"):
                    parts.append(f"Source: {data['AbstractSource']}")

            # Related topics
            topics = data.get("RelatedTopics", [])[:max_results]
            if topics and not parts:
                parts.append("Related information:")
                for topic in topics:
                    if "Text" in topic:
                        parts.append(f"  • {topic['Text'][:200]}")

            if parts:
                return "\n".join(parts)
            else:
                return f"No direct answer found for '{query}'. Try a more specific query."

        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {e}"

    # ── FETCH ─────────────────────────────────────────────────────────────────

    def fetch(self, url: str, max_chars: int = 3000) -> str:
        """
        Fetch a web page and extract clean readable text.
        Strips HTML tags, scripts, navigation clutter.
        Returns plain text suitable for LLM context.
        """
        try:
            r = self._client.get(url)

            if "text/html" not in r.headers.get("content-type", ""):
                return r.text[:max_chars]

            # Simple HTML cleaning — strip tags
            html = r.text

            # Remove script and style blocks entirely
            html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
            html = re.sub(r"<style[^>]*>.*?</style>",  " ", html, flags=re.DOTALL)
            html = re.sub(r"<nav[^>]*>.*?</nav>",      " ", html, flags=re.DOTALL)
            html = re.sub(r"<header[^>]*>.*?</header>","", html, flags=re.DOTALL)
            html = re.sub(r"<footer[^>]*>.*?</footer>","", html, flags=re.DOTALL)

            # Convert paragraph and heading tags to newlines
            html = re.sub(r"<br\s*/?>",        "\n", html)
            html = re.sub(r"<p[^>]*>",         "\n", html)
            html = re.sub(r"<h[1-6][^>]*>",    "\n\n", html)
            html = re.sub(r"</h[1-6]>",        "\n", html)
            html = re.sub(r"<li[^>]*>",        "\n  • ", html)

            # Strip remaining tags
            html = re.sub(r"<[^>]+>", " ", html)

            # Decode common entities
            html = html.replace("&amp;",  "&")
            html = html.replace("&lt;",   "<")
            html = html.replace("&gt;",   ">")
            html = html.replace("&quot;", '"')
            html = html.replace("&#39;",  "'")
            html = html.replace("&nbsp;", " ")

            # Collapse whitespace
            html = re.sub(r"[ \t]+",  " ",  html)
            html = re.sub(r"\n{3,}", "\n\n", html)

            text = html.strip()[:max_chars]
            return text if text else "Could not extract text from page."

        except httpx.TimeoutException:
            return f"Fetch timed out for {url}"
        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            return f"Could not fetch {url}: {e}"

    # ── WEATHER ───────────────────────────────────────────────────────────────

    def get_weather(self, location: str) -> str:
        """
        Get current weather + today's forecast for a location.
        Uses Open-Meteo (free, no API key, 10,000 calls/day).
        """
        try:
            # Step 1: geocode location name to coordinates
            geo_r = self._client.get(GEOCODE_URL, params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            })
            geo_data = geo_r.json()

            if not geo_data.get("results"):
                return f"Could not find location '{location}'."

            result = geo_data["results"][0]
            lat      = result["latitude"]
            lon      = result["longitude"]
            name     = result.get("name", location)
            country  = result.get("country", "")

            # Step 2: get weather data
            wx_r = self._client.get(WEATHER_URL, params={
                "latitude":           lat,
                "longitude":          lon,
                "current":            "temperature_2m,relative_humidity_2m,"
                                      "apparent_temperature,precipitation,"
                                      "wind_speed_10m,weathercode",
                "daily":              "temperature_2m_max,temperature_2m_min,"
                                      "precipitation_sum,weathercode",
                "timezone":           "auto",
                "forecast_days":      3,
                "wind_speed_unit":    "kmh",
            })
            wx = wx_r.json()

            current  = wx.get("current", {})
            daily    = wx.get("daily", {})
            units    = wx.get("current_units", {})

            temp     = current.get("temperature_2m", "?")
            feels    = current.get("apparent_temperature", "?")
            humidity = current.get("relative_humidity_2m", "?")
            wind     = current.get("wind_speed_10m", "?")
            precip   = current.get("precipitation", 0)
            code     = current.get("weathercode", 0)
            desc     = _wmo_code_to_description(code)

            lines = [
                f"Weather in {name}, {country}:",
                f"  {desc}, {temp}°C (feels like {feels}°C)",
                f"  Humidity: {humidity}%  Wind: {wind} km/h",
            ]

            if precip > 0:
                lines.append(f"  Precipitation: {precip}mm")

            # Next 3 days
            if daily.get("time"):
                lines.append("\nForecast:")
                for i, date in enumerate(daily["time"][:3]):
                    hi  = daily["temperature_2m_max"][i]
                    lo  = daily["temperature_2m_min"][i]
                    dc  = daily["weathercode"][i]
                    dd  = _wmo_code_to_description(dc)
                    lines.append(f"  {date}: {dd}, {lo}–{hi}°C")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Weather error for '{location}': {e}")
            return f"Could not get weather for '{location}': {e}"

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass


# ── WMO WEATHER CODE DESCRIPTIONS ────────────────────────────────────────────

def _wmo_code_to_description(code: int) -> str:
    descriptions = {
        0:  "Clear sky",
        1:  "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Weather code {code}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# python tools/web_tools.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tools = WebTools()

    print("\n── Search ──")
    print(tools.search("what is LangGraph"))

    print("\n── Weather ──")
    print(tools.get_weather("Deoghar, Jharkhand"))

    print("\n── Fetch ──")
    print(tools.fetch("https://example.com")[:500])
