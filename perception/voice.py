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
    ):
        self.sample_rate = sample_rate
        self.wake_words = wake_words or ["lavender", "hey lavender"]
        self.silence_threshold = silence_threshold  # seconds of silence = end of speech
        self.input_device = input_device

        # Internal state
        self._audio_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_listening_actively = False  # True = we heard the wake word, collect full utterance

        # Audio buffer for look-back (captures audio before wake word is processed)
        # Faster-Whisper has latency; we buffer so we don't miss the start of speech
        self._audio_buffer: list = []
        self._buffer_max_seconds = 6
        self._buffer_max_chunks = int(self._buffer_max_seconds * sample_rate / 4000)

        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # Use 4 threads for CPU, 1 for GPU
            cpu_threads=4 if device == "cpu" else 1,
        )
        logger.info("Whisper model loaded.")

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
            language="en",
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
        Main entry point. Call in a loop.
        Yields clean transcribed utterances (wake word stripped).

        Usage:
            for text in voice.listen():
                handle(text)

        The wake word can appear anywhere in the flow:
          - Passive mode: waiting for wake word in rolling chunks
          - After wake word: collects full utterance, transcribes, yields
          - If the full utterance is in the same chunk as the wake word, handles it
        """
        logger.info(f"Listening for wake words: {self.wake_words}")
        logger.info("Passive listen mode active.")

        if sd is None:
            logger.error("Sounddevice/PortAudio not available. Voice input disabled.")
            return

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=4000,         # ~0.25s per chunk
            device=self.input_device,
            callback=self._audio_callback,
        ):
            while not self._stop_event.is_set():

                # ── PASSIVE PHASE ──
                # Collect ~2 seconds of audio, quick-transcribe for wake word
                passive_audio = []
                passive_target_chunks = 8  # ~2 seconds

                for _ in range(passive_target_chunks):
                    try:
                        chunk = self._audio_queue.get(timeout=0.5)
                        passive_audio.append(chunk)
                    except queue.Empty:
                        continue

                if not passive_audio:
                    continue

                audio_np = np.concatenate(passive_audio)

                # Quick transcription — small audio, fast pass
                quick_text = self._transcribe(audio_np)

                if not quick_text:
                    continue

                if not self._contains_wake_word(quick_text):
                    logger.debug(f"No wake word in: '{quick_text}'")
                    continue

                # ── WAKE WORD DETECTED ──
                logger.info(f"Wake word detected in: '{quick_text}'")

                # Strip the wake word from whatever we already have
                initial_text = self._strip_wake_word(quick_text)

                # If there's already a full command in the same utterance, use it
                if len(initial_text.split()) >= 2:
                    logger.info(f"Full utterance captured with wake word: '{initial_text}'")
                    yield initial_text
                    continue

                # Otherwise, collect the rest of the utterance
                logger.info("Collecting remainder of utterance...")
                remainder_audio = self._collect_audio(max_seconds=12.0)

                if len(remainder_audio) < self.sample_rate * 0.3:
                    # Less than 0.3s of audio — probably nothing meaningful
                    logger.debug("Remainder too short, ignoring.")
                    continue

                remainder_text = self._transcribe(remainder_audio)

                # Combine initial + remainder (both stripped of wake word)
                full_text = (initial_text + " " + remainder_text).strip()
                full_text = self._strip_wake_word(full_text)  # Safety pass

                if full_text:
                    logger.info(f"Yielding: '{full_text}'")
                    yield full_text
                else:
                    logger.debug("Empty utterance after wake word, ignoring.")

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
