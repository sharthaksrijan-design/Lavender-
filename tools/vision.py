"""
LAVENDER — Vision
tools/vision.py

Gives Lavender eyes. She can see:
  - Documents you hold up to the camera
  - Your screen (screenshot capture)
  - The room via RealSense color stream
  - Images you paste into the session

Uses LLaVA (Large Language and Vision Assistant) running locally via Ollama.
No cloud vision API needed — everything stays on your machine.

Model: llava:13b (recommended) or llava:7b (faster, smaller GPU)
Download: ollama pull llava:13b

Usage from the brain:
  vision = Vision()
  result = vision.describe_image(path_or_bytes)
  result = vision.read_document(path_or_bytes)
  result = vision.capture_screen()
  result = vision.capture_camera()

The tool_registry registers vision as a LangGraph tool.
The brain routes PERCEPTUAL intents through it.
"""

import base64
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union
import httpx

logger = logging.getLogger("lavender.vision")

# Default vision model via Ollama
DEFAULT_VISION_MODEL = "llava:13b"

# Max image dimension before downscaling (LLaVA handles up to 336x336 natively,
# but modern versions accept larger; we cap to avoid VRAM spikes)
MAX_IMAGE_DIM = 1024


class Vision:
    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.model = model
        self.ollama_url = ollama_base_url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ── CORE VISION CALL ─────────────────────────────────────────────────────

    def _ask_vision(self, prompt: str, image_bytes: bytes) -> str:
        """
        Send an image + prompt to LLaVA via Ollama's /api/generate endpoint.
        Returns the model's text response.
        """
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        try:
            r = self._client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()

        except httpx.TimeoutException:
            return "Vision timed out. Try a smaller image or use llava:7b."
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return (f"Vision model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}")
            return f"Vision error: {e}"
        except Exception as e:
            logger.error(f"Vision call failed: {e}")
            return f"Vision unavailable: {e}"

    # ── IMAGE LOADING ─────────────────────────────────────────────────────────

    def _load_image(self, source: Union[str, bytes, Path]) -> Optional[bytes]:
        """
        Load image bytes from a file path, URL, or raw bytes.
        Optionally downscales large images before sending.
        """
        if isinstance(source, bytes):
            return self._maybe_resize(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return self._maybe_resize(path.read_bytes())
            # Treat as URL
            if str(source).startswith("http"):
                try:
                    r = self._client.get(str(source), timeout=10.0)
                    return self._maybe_resize(r.content)
                except Exception as e:
                    logger.error(f"Could not fetch image URL: {e}")
                    return None

        return None

    def _maybe_resize(self, image_bytes: bytes) -> bytes:
        """Resize image if it exceeds MAX_IMAGE_DIM to keep VRAM reasonable."""
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size

            if max(w, h) <= MAX_IMAGE_DIM:
                return image_bytes

            # Maintain aspect ratio
            scale = MAX_IMAGE_DIM / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            fmt = img.format or "JPEG"
            img.save(buf, format=fmt)
            logger.debug(f"Image resized: {w}x{h} → {new_w}x{new_h}")
            return buf.getvalue()

        except ImportError:
            # PIL not installed — pass image through as-is
            return image_bytes
        except Exception as e:
            logger.warning(f"Image resize failed: {e}")
            return image_bytes

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    def describe_image(
        self,
        source: Union[str, bytes, Path],
        context: str = "",
    ) -> str:
        """
        General-purpose image description.
        Answers 'what is in this image?' with detail.

        source:  File path, URL, or raw bytes
        context: Optional context hint (e.g., "this is a diagram from a talk")
        """
        image_bytes = self._load_image(source)
        if not image_bytes:
            return "Could not load the image."

        prompt = "Describe this image in detail."
        if context:
            prompt = f"Context: {context}\n\nDescribe this image in detail."

        return self._ask_vision(prompt, image_bytes)

    def read_document(
        self,
        source: Union[str, bytes, Path],
        question: str = "",
    ) -> str:
        """
        Extract and read text from a document image.
        Good for: handwritten notes, printed pages, whiteboards, receipts.

        question: If provided, answer this specific question about the document.
                  Otherwise, transcribe all visible text.
        """
        image_bytes = self._load_image(source)
        if not image_bytes:
            return "Could not load the document image."

        if question:
            prompt = (
                f"This is an image of a document or page. "
                f"Answer this question about it: {question}\n\n"
                f"If the answer is not visible in the image, say so clearly."
            )
        else:
            prompt = (
                "This is an image of a document, page, or written text. "
                "Transcribe all visible text exactly as it appears, "
                "preserving structure (headings, bullet points, tables). "
                "If text is unclear, indicate with [unclear]."
            )

        return self._ask_vision(prompt, image_bytes)

    def analyze_diagram(
        self,
        source: Union[str, bytes, Path],
        domain: str = "",
    ) -> str:
        """
        Analyze a technical diagram, chart, or architecture diagram.
        domain: optional hint like "software architecture", "circuit", "flowchart"
        """
        image_bytes = self._load_image(source)
        if not image_bytes:
            return "Could not load the diagram."

        domain_hint = f"This appears to be a {domain} diagram. " if domain else ""
        prompt = (
            f"{domain_hint}Analyze this diagram carefully. "
            "Describe: what it shows, the key components and their relationships, "
            "the flow or structure, and any notable patterns or issues you observe."
        )

        return self._ask_vision(prompt, image_bytes)

    def compare_images(
        self,
        source_a: Union[str, bytes, Path],
        source_b: Union[str, bytes, Path],
        question: str = "What are the key differences between these two images?",
    ) -> str:
        """
        Compare two images. LLaVA doesn't natively support multi-image in one call,
        so we describe each separately and synthesize.
        """
        desc_a = self.describe_image(source_a)
        desc_b = self.describe_image(source_b)

        # Use plain LLM for synthesis (no image needed for this step)
        synthesis_prompt = (
            f"Image A: {desc_a}\n\n"
            f"Image B: {desc_b}\n\n"
            f"Question: {question}"
        )

        # Return raw synthesis prompt — brain will handle with plain LLM
        return synthesis_prompt

    def capture_screen(
        self,
        display: int = 0,
        question: str = "What is on the screen?",
    ) -> str:
        """
        Take a screenshot and have LLaVA describe it.
        Useful for: 'what does my screen show?', 'what's the error on screen?'
        display: 0 = primary display
        """
        screenshot_bytes = self._take_screenshot(display)
        if not screenshot_bytes:
            return "Could not capture screen. Is a display connected?"

        prompt = (
            f"This is a screenshot of a computer screen. {question} "
            "Be specific about visible text, application names, errors, "
            "and what the user appears to be doing."
        )

        return self._ask_vision(prompt, screenshot_bytes)

    def capture_camera(
        self,
        question: str = "What do you see?",
        camera_index: int = 0,
    ) -> str:
        """
        Capture one frame from a camera and describe it.
        Uses RealSense color stream if available, otherwise OpenCV.
        """
        frame_bytes = self._capture_camera_frame(camera_index)
        if not frame_bytes:
            return "Could not capture from camera."

        return self._ask_vision(question, frame_bytes)

    def read_from_camera(self, question: str) -> str:
        """
        Point the camera at something and ask a question about it.
        Used when user says "look at this" or "what does this say?"
        """
        frame_bytes = self._capture_camera_frame()
        if not frame_bytes:
            return "Camera not available."

        prompt = (
            f"Look at what is in front of the camera. {question} "
            "Be specific and detailed."
        )
        return self._ask_vision(prompt, frame_bytes)

    # ── CAPTURE HELPERS ───────────────────────────────────────────────────────

    def _take_screenshot(self, display: int = 0) -> Optional[bytes]:
        """Take a screenshot using scrot, gnome-screenshot, or PIL."""
        # Try PIL/mss first (cross-platform)
        try:
            import mss
            import mss.tools
            import io
            with mss.mss() as sct:
                monitor = sct.monitors[display + 1] if display < len(sct.monitors) - 1 else sct.monitors[1]
                screenshot = sct.grab(monitor)
                from PIL import Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
        except ImportError:
            pass

        # Fallback: scrot (Linux)
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp = f.name
            subprocess.run(["scrot", "-q", "85", tmp],
                           capture_output=True, timeout=5)
            return Path(tmp).read_bytes()
        except Exception:
            pass

        return None

    def _capture_camera_frame(self, camera_index: int = 0) -> Optional[bytes]:
        """Capture one frame from camera. Tries RealSense, then OpenCV."""
        # Try RealSense first
        try:
            import pyrealsense2 as rs
            import numpy as np
            import io

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            try:
                # Discard a few frames for auto-exposure to settle
                for _ in range(5):
                    pipeline.wait_for_frames(timeout_ms=200)

                frames = pipeline.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if color_frame:
                    import cv2
                    img = np.asanyarray(color_frame.get_data())
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    return bytes(buf)
            finally:
                pipeline.stop()

        except Exception:
            pass

        # Fallback: OpenCV
        try:
            import cv2
            import io

            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return None

            # Let auto-exposure settle
            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            cap.release()

            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return bytes(buf)

        except Exception:
            pass

        return None

    def is_available(self) -> bool:
        """Check if the vision model is loaded in Ollama."""
        try:
            r = self._client.get(f"{self.ollama_url}/api/tags", timeout=3.0)
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass


# ── LANGRAPH TOOL WRAPPER ─────────────────────────────────────────────────────

def make_vision_tools(vision: Vision) -> list:
    """
    Returns a list of LangGraph tools wrapping the Vision class.
    Call build_toolkit() which calls this internally.
    """
    from langchain_core.tools import tool

    @tool
    def look_at_screen(question: str) -> str:
        """
        Take a screenshot and answer a question about what's on screen.
        Use when the user asks about their screen, an error, or visible content.
        Examples:
          look_at_screen("What error is shown?")
          look_at_screen("What application is open?")
          look_at_screen("Summarize what's on the screen")
        """
        return vision.capture_screen(question=question)

    @tool
    def look_at_camera(question: str) -> str:
        """
        Capture from the camera and answer a question about what it sees.
        Use when the user holds something up, points at something,
        or says 'look at this', 'what does this say', 'can you see this'.
        Examples:
          look_at_camera("What does this document say?")
          look_at_camera("What is this component?")
          look_at_camera("Read the text on this page")
        """
        return vision.read_from_camera(question)

    @tool
    def analyze_image_file(file_path: str, question: str = "") -> str:
        """
        Analyze an image file. Use when the user references a specific image path.
        file_path: absolute path to the image file
        question: optional specific question; if empty, describe the image generally
        """
        if question:
            return vision.read_document(file_path, question=question)
        return vision.describe_image(file_path)

    return [look_at_screen, look_at_camera, analyze_image_file]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# python tools/vision.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    vision = Vision()

    if not vision.is_available():
        print(f"Vision model '{vision.model}' not loaded.")
        print(f"Run: ollama pull {vision.model}")
        sys.exit(1)

    print(f"Vision model available: {vision.model}")

    print("\n── Screen capture test ──")
    result = vision.capture_screen(question="Describe what's on screen in one sentence.")
    print(result)

    print("\n── Camera test ──")
    result = vision.capture_camera(question="What do you see in front of the camera?")
    print(result)
