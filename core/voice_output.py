"""
LAVENDER — Voice Output
core/voice_output.py

Handles all audio output:
  - ElevenLabs TTS with per-personality voice models and settings
  - Audio playback via sounddevice
  - Interrupt support — stop mid-sentence when new input arrives
  - Volume control
  - Graceful fallback if ElevenLabs is unavailable (pyttsx3 offline TTS)
"""

import io
import os
import threading
import logging
import numpy as np
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
import soundfile as sf
from typing import Optional

logger = logging.getLogger("lavender.voice_output")


# ── PERSONALITY VOICE SETTINGS ────────────────────────────────────────────────
# These are applied to ElevenLabs generation per personality.
# stability: how consistent the voice is (0.0 = variable, 1.0 = robotic)
# similarity: how close to the cloned voice (higher = more accurate but less expressive)
# style: emotional expressiveness (0.0 = neutral, 1.0 = maximum)

PERSONALITY_VOICE_SETTINGS = {
    "iris": {
        "stability": 0.85,
        "similarity_boost": 0.80,
        "style": 0.10,
        "use_speaker_boost": True,
    },
    "nova": {
        "stability": 0.70,
        "similarity_boost": 0.85,
        "style": 0.50,
        "use_speaker_boost": True,
    },
    "vector": {
        "stability": 0.80,
        "similarity_boost": 0.82,
        "style": 0.20,
        "use_speaker_boost": True,
    },
    "solace": {
        "stability": 0.90,
        "similarity_boost": 0.88,
        "style": 0.30,
        "use_speaker_boost": True,
    },
    "lilac": {
        "stability": 0.75,
        "similarity_boost": 0.90,
        "style": 0.60,   # More expressive — she means what she says
        "use_speaker_boost": True,
    },
}

# Lilac pauses before her roast+answer to let the roast land.
# Format: (roast_audio, pause_seconds, answer_audio)
LILAC_ANSWER_PAUSE = 1.2  # seconds of silence between Lilac's roast and her answer


class VoiceOutput:
    def __init__(
        self,
        api_key: str,
        voice_ids: dict[str, str],
        output_device=None,
        volume: float = 1.0,
        model: str = "eleven_multilingual_v2",
    ):
        """
        api_key:      ElevenLabs API key
        voice_ids:    dict mapping personality name → ElevenLabs voice ID
                      e.g. {"iris": "abc123", "nova": "def456", ...}
        output_device: sounddevice device index or name (None = system default)
        volume:       output volume multiplier (0.0 - 2.0)
        model:        ElevenLabs model ID
        """
        self.api_key      = api_key
        self.voice_ids    = voice_ids
        self.output_device = output_device
        self.volume       = volume
        self.model        = model
        self.current_personality = "nova"

        # Interrupt control
        self._stop_event  = threading.Event()
        self._speaking    = False
        self._speak_lock  = threading.Lock()

        # Initialize ElevenLabs client
        self._client = None
        if api_key and api_key != "your_api_key_here":
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=api_key)
                logger.info("ElevenLabs client initialized.")
            except ImportError:
                logger.warning("elevenlabs package not installed. Using fallback TTS.")
            except Exception as e:
                logger.warning(f"ElevenLabs init failed: {e}. Using fallback TTS.")

        if not self._client:
            logger.warning("Running without ElevenLabs. Using pyttsx3 fallback.")
            self._init_fallback_tts()

    def _init_fallback_tts(self):
        """Initialize pyttsx3 as offline fallback."""
        try:
            import pyttsx3
            self._fallback_tts = pyttsx3.init()
            self._fallback_tts.setProperty("rate", 170)   # words per minute
            self._fallback_tts.setProperty("volume", self.volume)
            logger.info("pyttsx3 fallback TTS initialized.")
        except Exception as e:
            self._fallback_tts = None
            logger.error(f"Fallback TTS also failed: {e}. No audio output available.")

    # ── PERSONALITY ───────────────────────────────────────────────────────────

    def set_personality(self, personality: str):
        """Switch to a different personality's voice settings."""
        self.current_personality = personality
        logger.info(f"Voice personality set to: {personality}")

    # ── INTERRUPT ────────────────────────────────────────────────────────────

    def interrupt(self):
        """
        Stop any currently playing audio immediately.
        Called when the user speaks while Lavender is talking.
        """
        if self._speaking:
            logger.info("Voice output interrupted.")
            self._stop_event.set()
            sd.stop()  # Halt sounddevice playback

    def is_speaking(self) -> bool:
        return self._speaking

    # ── CORE SPEAK ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Synthesizes and plays audio for the given text.
        Blocks until playback is complete or interrupted.

        Handles the Lilac roast format: if text contains the roast+answer
        separator "\n\n...anyway.", it splits and pauses between them.
        """
        if not text or not text.strip():
            return

        # Handle Lilac's roast + answer format
        if "\n\n...anyway." in text:
            parts = text.split("\n\n...anyway.", 1)
            roast_text  = parts[0].strip()
            answer_text = ("...anyway. " + parts[1]).strip()

            self._play_text(roast_text)
            if not self._stop_event.is_set():
                self._pause(LILAC_ANSWER_PAUSE)
            if not self._stop_event.is_set():
                self._play_text(answer_text)
        else:
            self._play_text(text)

    def _play_text(self, text: str):
        """
        Internal: synthesize and play a single text string.
        """
        self._stop_event.clear()

        with self._speak_lock:
            self._speaking = True
            try:
                if self._client:
                    self._speak_elevenlabs(text)
                else:
                    self._speak_fallback(text)
            finally:
                self._speaking = False

    def _pause(self, seconds: float):
        """Silent pause, interruptible."""
        import time
        step = 0.05
        elapsed = 0.0
        while elapsed < seconds and not self._stop_event.is_set():
            time.sleep(step)
            elapsed += step

    # ── ELEVENLABS PLAYBACK ───────────────────────────────────────────────────

    def _speak_elevenlabs(self, text: str):
        """Generate audio via ElevenLabs and play it."""
        personality = self.current_personality
        voice_id    = self.voice_ids.get(personality)

        if not voice_id or voice_id == "voice_id_here":
            logger.warning(
                f"No voice ID configured for personality '{personality}'. "
                "Set VOICE_{PERSONALITY} in config/.env"
            )
            self._speak_fallback(text)
            return

        voice_settings = PERSONALITY_VOICE_SETTINGS.get(
            personality, PERSONALITY_VOICE_SETTINGS["nova"]
        )

        try:
            from elevenlabs import VoiceSettings

            logger.debug(f"Generating TTS: '{text[:60]}...' (voice: {voice_id})")

            audio_generator = self._client.generate(
                text=text,
                voice=voice_id,
                voice_settings=VoiceSettings(**voice_settings),
                model=self.model,
            )

            # Collect all audio bytes
            audio_bytes = b"".join(audio_generator)

            if self._stop_event.is_set():
                return

            # Decode and play
            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32"
            )

            # Apply volume
            audio_data = audio_data * self.volume

            self._play_audio(audio_data, sample_rate)

        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
            logger.info("Falling back to pyttsx3.")
            self._speak_fallback(text)

    # ── AUDIO PLAYBACK ────────────────────────────────────────────────────────

    def _play_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Plays a numpy array through sounddevice.
        Checks stop_event every 100ms to support interruption.
        """
        if self._stop_event.is_set():
            return

        try:
            if sd:
                # Play the entire array at once to avoid stuttering.
                # sd.play() is non-blocking.
                sd.play(audio_data, samplerate=sample_rate, device=self.output_device)

                # Periodically check for interruption while playing
                duration = len(audio_data) / sample_rate
                start_time = time.time()
                while time.time() - start_time < duration:
                    if self._stop_event.is_set():
                        sd.stop()
                        break
                    time.sleep(0.1)
                sd.wait() # Ensure playback is actually finished
            else:
                logger.error("Sounddevice not available for playback.")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    # ── FALLBACK TTS ──────────────────────────────────────────────────────────

    def _speak_fallback(self, text: str):
        """Use pyttsx3 if ElevenLabs is unavailable."""
        if hasattr(self, "_fallback_tts") and self._fallback_tts:
            try:
                self._fallback_tts.say(text)
                self._fallback_tts.runAndWait()
            except Exception as e:
                logger.error(f"Fallback TTS failed: {e}")
        else:
            # Absolute last resort: just print
            print(f"[AUDIO UNAVAILABLE] {text}")

    # ── VOLUME ────────────────────────────────────────────────────────────────

    def set_volume(self, volume: float):
        """Set output volume. 1.0 = normal, 0.0 = silent, 2.0 = double."""
        self.volume = max(0.0, min(2.0, volume))
        if hasattr(self, "_fallback_tts") and self._fallback_tts:
            self._fallback_tts.setProperty("volume", min(1.0, self.volume))
        logger.info(f"Volume set to {self.volume:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# Run: python core/voice_output.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_ids = {
        "iris":   os.getenv("VOICE_IRIS", ""),
        "nova":   os.getenv("VOICE_NOVA", ""),
        "vector": os.getenv("VOICE_VECTOR", ""),
        "solace": os.getenv("VOICE_SOLACE", ""),
        "lilac":  os.getenv("VOICE_LILAC", ""),
    }

    vo = VoiceOutput(api_key=api_key, voice_ids=voice_ids)

    test_lines = [
        ("nova",   "Hello. Nova online. Warm, curious, and ready."),
        ("iris",   "Iris. Online."),
        ("vector", "Vector ready. Give me something to solve."),
        ("solace", "I'm here. Take your time."),
        ("lilac",  "Lilac. What do you need. Make it worth my time."),
    ]

    print("\nTesting all five personality voices...\n")
    for personality, line in test_lines:
        print(f"[{personality.upper()}]: {line}")
        vo.set_personality(personality)
        vo.speak(line)
        import time; time.sleep(0.5)

    print("\nAll voices tested.")
