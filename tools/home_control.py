"""
LAVENDER — Home Control
tools/home_control.py

Wraps the Home Assistant REST API.
Lavender calls methods here to control lights, scenes, media,
climate, locks — anything connected to Home Assistant.

Setup:
  1. Install Home Assistant (https://www.home-assistant.io/)
     Easiest: run it on a Raspberry Pi or as a VM.
  2. Create a Long-Lived Access Token:
     HA UI → Profile → Long-Lived Access Tokens → Create Token
  3. Add HA_URL and HA_TOKEN to config/.env

Entity ID format: domain.entity_name
  e.g. light.desk_lamp, switch.monitor, climate.office_ac
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger("lavender.home_control")


class HomeControl:
    def __init__(self, ha_url: str, token: str, timeout: float = 5.0):
        self.ha_url = ha_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # Cache of known entities: entity_id → {state, attributes}
        self._entity_cache: dict = {}

    # ── CONNECTIVITY CHECK ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Home Assistant is reachable."""
        try:
            r = httpx.get(
                f"{self.ha_url}/api/",
                headers=self._headers,
                timeout=2.0
            )
            return r.status_code == 200
        except Exception:
            return False

    # ── STATE READING ─────────────────────────────────────────────────────────

    def get_state(self, entity_id: str) -> Optional[dict]:
        """Get the current state of an entity."""
        try:
            r = httpx.get(
                f"{self.ha_url}/api/states/{entity_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                data = r.json()
                self._entity_cache[entity_id] = data
                return data
            else:
                logger.warning(f"State fetch failed for {entity_id}: {r.status_code}")
                return None
        except Exception as e:
            logger.error(f"get_state error: {e}")
            return None

    def get_all_states(self) -> list[dict]:
        """Get all entity states — used for room awareness."""
        try:
            r = httpx.get(
                f"{self.ha_url}/api/states",
                headers=self._headers,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                states = r.json()
                # Update cache
                for s in states:
                    self._entity_cache[s["entity_id"]] = s
                return states
            return []
        except Exception as e:
            logger.error(f"get_all_states error: {e}")
            return []

    def get_entities_by_domain(self, domain: str) -> list[dict]:
        """Get all entities of a given domain (light, switch, climate, etc.)."""
        all_states = self.get_all_states()
        return [s for s in all_states if s["entity_id"].startswith(f"{domain}.")]

    def get_friendly_state(self, entity_id: str) -> str:
        """Returns a human-readable state string for an entity."""
        state = self.get_state(entity_id)
        if not state:
            return f"unknown (could not reach {entity_id})"

        s = state.get("state", "unknown")
        attrs = state.get("attributes", {})
        name = attrs.get("friendly_name", entity_id)

        # Domain-specific formatting
        domain = entity_id.split(".")[0]
        if domain == "light":
            if s == "on":
                brightness = attrs.get("brightness", 255)
                pct = round(brightness / 255 * 100)
                return f"{name} is on at {pct}% brightness"
            return f"{name} is off"

        if domain == "climate":
            temp = attrs.get("current_temperature", "?")
            target = attrs.get("temperature", "?")
            mode = attrs.get("hvac_mode", s)
            return f"{name} is {mode}, current {temp}°, target {target}°"

        if domain == "media_player":
            if s == "playing":
                title = attrs.get("media_title", "something")
                artist = attrs.get("media_artist", "")
                return f"{name} is playing {title}" + (f" by {artist}" if artist else "")
            return f"{name} is {s}"

        return f"{name} is {s}"

    # ── SERVICE CALLS ─────────────────────────────────────────────────────────

    def _call_service(self, domain: str, service: str, data: dict) -> bool:
        """
        Generic service call.
        Returns True on success, False on failure.
        """
        try:
            r = httpx.post(
                f"{self.ha_url}/api/services/{domain}/{service}",
                headers=self._headers,
                json=data,
                timeout=self.timeout,
            )
            success = r.status_code in (200, 201)
            if not success:
                logger.warning(f"Service call {domain}.{service} failed: {r.status_code}")
            return success
        except Exception as e:
            logger.error(f"Service call error: {e}")
            return False

    # ── LIGHTS ────────────────────────────────────────────────────────────────

    def turn_on(self, entity_id: str) -> bool:
        domain = entity_id.split(".")[0]
        return self._call_service(domain, "turn_on", {"entity_id": entity_id})

    def turn_off(self, entity_id: str) -> bool:
        domain = entity_id.split(".")[0]
        return self._call_service(domain, "turn_off", {"entity_id": entity_id})

    def toggle(self, entity_id: str) -> bool:
        domain = entity_id.split(".")[0]
        return self._call_service(domain, "toggle", {"entity_id": entity_id})

    def set_light(
        self,
        entity_id: str,
        brightness_pct: Optional[int] = None,
        color_temp_kelvin: Optional[int] = None,
        rgb_color: Optional[tuple] = None,
        transition: float = 1.0,
    ) -> bool:
        """
        Set light brightness, color temperature, or RGB color.
        brightness_pct: 0–100
        color_temp_kelvin: warm ~2700, neutral ~4000, cool ~6500
        rgb_color: (r, g, b) each 0–255
        """
        data: dict = {"entity_id": entity_id, "transition": transition}

        if brightness_pct is not None:
            data["brightness_pct"] = max(0, min(100, brightness_pct))

        if color_temp_kelvin is not None:
            data["color_temp_kelvin"] = color_temp_kelvin

        if rgb_color is not None:
            data["rgb_color"] = list(rgb_color)

        return self._call_service("light", "turn_on", data)

    # ── SCENES ────────────────────────────────────────────────────────────────

    def activate_scene(self, scene_id: str) -> bool:
        """Activate a saved HA scene. entity_id format: scene.scene_name"""
        if not scene_id.startswith("scene."):
            scene_id = f"scene.{scene_id}"
        return self._call_service("scene", "turn_on", {"entity_id": scene_id})

    def list_scenes(self) -> list[str]:
        """Return friendly names of available scenes."""
        scenes = self.get_entities_by_domain("scene")
        return [
            s.get("attributes", {}).get("friendly_name", s["entity_id"])
            for s in scenes
        ]

    # ── CLIMATE ───────────────────────────────────────────────────────────────

    def set_temperature(self, entity_id: str, temperature: float) -> bool:
        return self._call_service("climate", "set_temperature", {
            "entity_id": entity_id,
            "temperature": temperature,
        })

    def set_hvac_mode(self, entity_id: str, mode: str) -> bool:
        """mode: cool, heat, heat_cool, off, auto, fan_only, dry"""
        return self._call_service("climate", "set_hvac_mode", {
            "entity_id": entity_id,
            "hvac_mode": mode,
        })

    # ── MEDIA ────────────────────────────────────────────────────────────────

    def media_play_pause(self, entity_id: str) -> bool:
        return self._call_service("media_player", "media_play_pause",
                                  {"entity_id": entity_id})

    def media_stop(self, entity_id: str) -> bool:
        return self._call_service("media_player", "media_stop",
                                  {"entity_id": entity_id})

    def set_volume(self, entity_id: str, volume: float) -> bool:
        """volume: 0.0–1.0"""
        return self._call_service("media_player", "volume_set", {
            "entity_id": entity_id,
            "volume_level": max(0.0, min(1.0, volume)),
        })

    # ── NATURAL LANGUAGE RESOLVER ─────────────────────────────────────────────

    def resolve_entity(self, natural_name: str, domain: str = None) -> Optional[str]:
        """
        Resolve a natural language name to an entity ID.
        "desk lamp" → "light.desk_lamp"
        "office AC" → "climate.office_ac"

        Uses the entity cache (populate via get_all_states first).
        """
        natural_lower = natural_name.lower().strip()

        candidates = list(self._entity_cache.values())
        if domain:
            candidates = [e for e in candidates
                         if e["entity_id"].startswith(f"{domain}.")]

        # Exact friendly name match
        for entity in candidates:
            friendly = entity.get("attributes", {}).get("friendly_name", "").lower()
            if friendly == natural_lower:
                return entity["entity_id"]

        # Partial match
        for entity in candidates:
            friendly = entity.get("attributes", {}).get("friendly_name", "").lower()
            entity_id = entity["entity_id"].lower()
            if natural_lower in friendly or natural_lower in entity_id:
                return entity["entity_id"]

        return None

    def execute_natural_command(self, command: str) -> str:
        """
        Execute a structured command dict from the LLM tool call.
        command format: "turn on desk lamp" / "set temperature to 22" / etc.

        This is called by the LangGraph tool — returns a string result
        for the LLM to include in its response.
        """
        import json

        # command is expected to be a JSON string from the LLM tool call
        try:
            cmd = json.loads(command)
        except (json.JSONDecodeError, TypeError):
            return f"Could not parse command: {command}"

        action     = cmd.get("action", "").lower()
        entity     = cmd.get("entity", "")
        value      = cmd.get("value")
        domain     = cmd.get("domain")

        # Try to resolve entity name to ID
        entity_id = entity if "." in entity else self.resolve_entity(entity, domain)

        if not entity_id:
            # Try refreshing cache and resolving again
            self.get_all_states()
            entity_id = entity if "." in entity else self.resolve_entity(entity, domain)

        if not entity_id:
            return f"Could not find '{entity}' in Home Assistant. " \
                   f"Check entity name or run 'Lavender, what devices do I have?'"

        # Execute action
        if action in ("turn on", "on", "enable"):
            success = self.turn_on(entity_id)
            return f"Turned on {entity_id}." if success else f"Failed to turn on {entity_id}."

        elif action in ("turn off", "off", "disable"):
            success = self.turn_off(entity_id)
            return f"Turned off {entity_id}." if success else f"Failed to turn off {entity_id}."

        elif action in ("toggle",):
            success = self.toggle(entity_id)
            return f"Toggled {entity_id}." if success else f"Failed to toggle {entity_id}."

        elif action in ("set brightness", "brightness"):
            pct = int(value) if value else 50
            success = self.set_light(entity_id, brightness_pct=pct)
            return f"Set {entity_id} to {pct}% brightness." if success else "Failed."

        elif action in ("set temperature", "temperature"):
            temp = float(value) if value else 22.0
            success = self.set_temperature(entity_id, temp)
            return f"Set {entity_id} to {temp}°." if success else "Failed."

        elif action in ("set volume", "volume"):
            vol = float(value) / 100.0 if float(value or 50) > 1 else float(value or 0.5)
            success = self.set_volume(entity_id, vol)
            return f"Set volume to {int(vol*100)}%." if success else "Failed."

        elif action in ("activate scene", "scene"):
            success = self.activate_scene(entity_id)
            return f"Scene '{entity_id}' activated." if success else "Failed."

        elif action in ("status", "state", "check"):
            return self.get_friendly_state(entity_id)

        else:
            return f"Unknown action '{action}'. Try: turn on/off, toggle, " \
                   f"set brightness, set temperature, status."


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# python tools/home_control.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    load_dotenv("config/.env")

    ha = HomeControl(
        ha_url=os.getenv("HA_URL", "http://homeassistant.local:8123"),
        token=os.getenv("HA_TOKEN", ""),
    )

    if not ha.is_available():
        print("Home Assistant not reachable. Check HA_URL and HA_TOKEN in config/.env")
    else:
        print("Connected to Home Assistant.\n")

        print("All lights:")
        for entity in ha.get_entities_by_domain("light"):
            print(f"  {entity['entity_id']}: {entity['state']}")

        print("\nAll scenes:")
        for scene in ha.list_scenes():
            print(f"  {scene}")
