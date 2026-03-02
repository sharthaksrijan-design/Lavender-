"""
LAVENDER — Voice Perception
perception/voice.py

Handles all microphone input:
  - Continuous listening in a background thread
  - Wake word detection (lightweight, runs always)
  - Speech-to-text via Faster-Whisper (CUDA-accelerated)
  - Voice activity detection to know when speech ends
  - Yields clean transcribed text to the main loop
"""

import queue
import threading
import time
import numpy as np
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
from faster_whisper import WhisperModel
import openwakeword
from openwakeword.model import Model
from typing import Generator, Optional
import logging

logger = logging.getLogger("lavender.voice")


class VoicePerception:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        wake_words: list[str] = None,
        silence_threshold: float = 1.2,
        input_device=None,
        language: str = "en",
    ):
        self.sample_rate = sample_rate
        self.wake_words = wake_words or ["lavender", "hey lavender"]
        self.silence_threshold = silence_threshold  # seconds of silence = end of speech
        self.input_device = input_device
        self.language = language

        # Internal state
        self._audio_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_listening_actively = False  # True = we heard the wake word, collect full utterance
        self._last_interaction_time = 0
        self._conversation_window = 30.0 # 30s follow-up window

        # Audio buffer for look-back (captures audio before wake word is processed)
        # Faster-Whisper has latency; we buffer so we don't miss the start of speech
        self._audio_buffer: list = []
        self._buffer_max_seconds = 6
        self._buffer_max_chunks = int(self._buffer_max_seconds * sample_rate / 1280) # openWakeWord uses 1280 samples

        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # Use 4 threads for CPU, 1 for GPU
            cpu_threads=4 if device == "cpu" else 1,
        )
        logger.info("Whisper model loaded.")

        # Initialize openWakeWord
        logger.info("Loading openWakeWord model...")
        self.oww_model = Model(
            wakeword_models=["hey_jarvis", "alexa"], # placeholders or custom models
            inference_framework="onnx"
        )
        logger.info("openWakeWord ready.")

    # ── AUDIO CALLBACK ───────────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Called by sounddevice for every audio block.
        We copy the block into the queue and also maintain a rolling buffer.
        """
        if status:
            logger.warning(f"Audio status: {status}")

        chunk = indata.copy().flatten().astype(np.float32)
        self._audio_queue.put(chunk)

        # Maintain rolling buffer of recent audio for look-back
        self._audio_buffer.append(chunk)
        if len(self._audio_buffer) > self._buffer_max_chunks:
            self._audio_buffer.pop(0)

    # ── TRANSCRIPTION ────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Runs Faster-Whisper on a numpy float32 audio array.
        Returns the transcribed string, or empty string if nothing detected.
        """
        segments, info = self.model.transcribe(
            audio,
            language=self.language, # Language lock
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(self.silence_threshold * 1000),
                "threshold": 0.4,
            },
            beam_size=5,
            without_timestamps=True,
        )

        text = " ".join(segment.text for segment in segments).strip()
        logger.debug(f"Transcribed: '{text}'")
        return text

    # ── WAKE WORD CHECK ──────────────────────────────────────────────────────

    def _contains_wake_word(self, text: str) -> bool:
        text_lower = text.lower().strip()
        return any(wake_word in text_lower for wake_word in self.wake_words)

    def _strip_wake_word(self, text: str) -> str:
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                # Remove wake word and any trailing comma/space
                remainder = text[len(wake_word):].lstrip(", ").strip()
                return remainder
        return text.strip()

    # ── AUDIO COLLECTION ─────────────────────────────────────────────────────

    def _collect_audio(self, max_seconds: float = 15.0) -> np.ndarray:
        """
        Collects audio chunks from the queue until silence is detected
        or max_seconds is reached.

        This is called after a wake word is detected to capture the
        full user utterance.
        """
        collected = []
        silence_chunks = 0
        # How many chunks of silence = end of speech
        # chunk size = 4000 samples at 16kHz = 0.25 seconds
        chunks_per_second = self.sample_rate / 4000
        silence_chunk_limit = int(self.silence_threshold * chunks_per_second)
        max_chunks = int(max_seconds * chunks_per_second)
        rms_silence_threshold = 0.005  # RMS below this = silence

        start_time = time.time()

        while len(collected) < max_chunks:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                collected.append(chunk)

                # Check RMS energy to detect silence
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < rms_silence_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0  # Reset on any speech

                if silence_chunks >= silence_chunk_limit and len(collected) > 10:
                    # We have enough audio and detected silence — utterance is done
                    break

            except queue.Empty:
                if time.time() - start_time > max_seconds:
                    break

        if not collected:
            return np.array([], dtype=np.float32)

        return np.concatenate(collected)

    # ── MAIN GENERATOR ───────────────────────────────────────────────────────

    def listen(self) -> Generator[str, None, None]:
        """
        Main entry point. Optimized with openWakeWord and Conversation Mode.
        """
        logger.info(f"Listening with openWakeWord: {self.wake_words}")

        if sd is None:
            logger.error("Sounddevice/PortAudio not available. Voice input disabled.")
            return

        # openWakeWord requires 16kHz, mono, 16-bit PCM or 32-bit float
        # Chunk size for OWW is usually 1280 samples (80ms at 16kHz)
        chunk_size = 1280

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            device=self.input_device,
            callback=self._audio_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # ── CONVERSATION MODE CHECK ──
                # If we're within the 30s window, skip wake word and go straight to collection
                in_conversation = (time.time() - self._last_interaction_time) < self._conversation_window

                if in_conversation:
                    # Collect and transcribe immediately
                    logger.info("Conversation mode: active.")
                    utterance_audio = self._collect_audio(max_seconds=12.0)
                    text = self._transcribe(utterance_audio)
                    if text:
                        self._last_interaction_time = time.time()
                        yield text
                    continue

                # ── PASSIVE PHASE (openWakeWord) ──
                # Feed chunk to OWW
                prediction = self.oww_model.predict(chunk)

                # Check all active models
                wake_word_triggered = False
                for model_name, score in prediction.items():
                    if score > 0.5: # Threshold
                        logger.info(f"Wake word '{model_name}' detected (score: {score:.2f})")
                        wake_word_triggered = True
                        break

                if wake_word_triggered:
                    # ── WAKE WORD DETECTED ──
                    logger.info("Collecting utterance...")
                    utterance_audio = self._collect_audio(max_seconds=12.0)
                    text = self._transcribe(utterance_audio)
                    if text:
                        self._last_interaction_time = time.time()
                        yield text

    def stop(self):
        """Signal the listen loop to stop."""
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# Run: python perception/voice.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s"
    )

    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print("\nStarting voice perception test.")
    print(f"Say 'Lavender' followed by anything.\n")

    vp = VoicePerception(
        model_size="medium",
        device="cuda" if "--cpu" not in sys.argv else "cpu",
        compute_type="float16" if "--cpu" not in sys.argv else "int8",
    )

    for utterance in vp.listen():
        print(f"\n[You said]: {utterance}\n")
