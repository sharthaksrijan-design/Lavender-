"""
LAVENDER — Gesture Zone Calibration
scripts/calibrate_zones.py

Interactive tool to calibrate the three interaction zones to your
actual table geometry. Run this once after setting up the RealSense.

Steps:
  1. Place your hand at each zone boundary when prompted
  2. The script records depth readings and computes bounds
  3. Saves calibrated values to config/zone_calibration.yaml
  4. gesture.py reads this file on startup if it exists

Usage:
  python scripts/calibrate_zones.py
  python scripts/calibrate_zones.py --preview   (show live depth feed)
"""

import sys
import time
import yaml
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate")

OUTPUT_PATH = ROOT / "config" / "zone_calibration.yaml"


def mean_depth_at_hand(pipeline, n_samples=20) -> tuple[float, float, float]:
    """
    Capture N frames, return (x, y, z) centroid of the closest detected hand region.
    Simplified: returns average of nearest 10% of depth pixels in center region.
    """
    try:
        import pyrealsense2 as rs
        import numpy as np

        depths = []
        for _ in range(n_samples):
            frames = pipeline.wait_for_frames(timeout_ms=500)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            h, w = depth_image.shape

            # Center crop — ignore edges
            cx1, cx2 = w // 4, 3 * w // 4
            cy1, cy2 = h // 4, 3 * h // 4
            crop = depth_image[cy1:cy2, cx1:cx2].astype(float)

            # Convert to meters
            depth_scale = (pipeline.get_active_profile()
                           .get_device().first_depth_sensor()
                           .get_depth_scale())
            crop_m = crop * depth_scale

            # Remove zeros (no depth reading)
            valid = crop_m[crop_m > 0.1]
            if len(valid) == 0:
                continue

            # Find hand: nearest 10% of points
            threshold = np.percentile(valid, 10)
            hand_pixels = crop_m[crop_m < threshold + 0.05]
            if len(hand_pixels):
                depths.append(float(np.median(hand_pixels)))

        if not depths:
            return (0, 0, 0)

        z = float(np.median(depths))
        # Approximate x, y from center of frame — good enough for calibration
        return (0.0, 0.0, z)

    except Exception as e:
        logger.error(f"Depth measurement failed: {e}")
        return (0, 0, 0)


def calibrate():
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed. Run: pip install pyrealsense2")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  LAVENDER — Zone Calibration")
    print("─" * 60)
    print()
    print("  This tool measures the depth positions of your interaction")
    print("  zones so gesture recognition works with your exact setup.")
    print()
    print("  You'll be asked to place your hand at each zone boundary.")
    print("  Hold still for 3 seconds when prompted, then press Enter.")
    print()

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        print("  RealSense connected. Warming up...")
        time.sleep(2)
        print()

        readings = {}

        # ── PANEL ZONE ────────────────────────────────────────────────────────
        print("  ── PANEL ZONE ──────────────────────────────────────────────")
        print()

        input("  1. Place your hand at the NEAR edge of the panel zone\n"
              "     (closest to you, where you'd first reach to touch the panel)\n"
              "     Hold still. Press Enter when ready... ")
        print("  Measuring...", end="", flush=True)
        _, _, z = mean_depth_at_hand(pipeline)
        readings["panel_z_near"] = round(z, 3)
        print(f" {z:.3f}m")

        input("\n  2. Place your hand at the FAR edge of the panel zone\n"
              "     (furthest away — back of the display area)\n"
              "     Hold still. Press Enter when ready... ")
        print("  Measuring...", end="", flush=True)
        _, _, z = mean_depth_at_hand(pipeline)
        readings["panel_z_far"] = round(z, 3)
        print(f" {z:.3f}m")

        input("\n  3. Place your fingertip TOUCHING the panel surface\n"
              "     (the holographic film / projection surface itself)\n"
              "     Hold still. Press Enter when ready... ")
        print("  Measuring...", end="", flush=True)
        _, _, z = mean_depth_at_hand(pipeline)
        readings["panel_plane"] = round(z, 3)
        print(f" {z:.3f}m")

        # ── FOG ZONE ──────────────────────────────────────────────────────────
        print()
        print("  ── FOG ZONE ─────────────────────────────────────────────────")
        print()

        yn = input("  Do you have a fog well? [y/N]: ").strip().lower()
        if yn == "y":
            input("  4. Place your hand at the NEAR edge of the fog zone\n"
                  "     (near edge of the fog well)\n"
                  "     Hold still. Press Enter when ready... ")
            print("  Measuring...", end="", flush=True)
            _, _, z = mean_depth_at_hand(pipeline)
            readings["fog_z_near"] = round(z, 3)
            print(f" {z:.3f}m")

            input("\n  5. Place your hand at the FAR edge of the fog zone\n"
                  "     Hold still. Press Enter when ready... ")
            print("  Measuring...", end="", flush=True)
            _, _, z = mean_depth_at_hand(pipeline)
            readings["fog_z_far"] = round(z, 3)
            print(f" {z:.3f}m")
        else:
            # Use defaults offset from panel
            panel_mid = (readings.get("panel_z_near", 0.4) +
                         readings.get("panel_z_far", 0.7)) / 2
            readings["fog_z_near"] = round(panel_mid + 0.05, 3)
            readings["fog_z_far"]  = round(panel_mid + 0.4, 3)
            print(f"  Using defaults: {readings['fog_z_near']}m – {readings['fog_z_far']}m")

    finally:
        pipeline.stop()

    # ── BUILD CALIBRATION DICT ────────────────────────────────────────────────
    margin = 0.03  # 3cm margin

    calibration = {
        "panel": {
            "z_min": min(readings["panel_z_near"], readings["panel_z_far"]) - margin,
            "z_max": max(readings["panel_z_near"], readings["panel_z_far"]) + margin,
            "z_plane": readings["panel_plane"],
            "x_min": -0.40,
            "x_max":  0.40,
            "y_min": -0.25,
            "y_max":  0.35,
        },
        "fog": {
            "z_min": min(readings["fog_z_near"], readings["fog_z_far"]) - margin,
            "z_max": max(readings["fog_z_near"], readings["fog_z_far"]) + margin,
            "x_min": -0.60,
            "x_max": -0.10,
            "y_min": -0.15,
            "y_max":  0.45,
        },
    }

    # ── SAVE ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump({"zones": calibration}, f, default_flow_style=False)

    print()
    print("─" * 60)
    print(f"  Calibration saved to: {OUTPUT_PATH}")
    print()
    print("  Zone bounds:")
    print(f"  Panel: z {calibration['panel']['z_min']:.3f} – {calibration['panel']['z_max']:.3f}m")
    print(f"         panel plane at {calibration['panel']['z_plane']:.3f}m")
    print(f"  Fog:   z {calibration['fog']['z_min']:.3f} – {calibration['fog']['z_max']:.3f}m")
    print()
    print("  Restart Lavender to apply.")
    print("─" * 60)
    print()


def preview():
    """Show live depth feed with zone overlay for visual inspection."""
    try:
        import pyrealsense2 as rs
        import numpy as np
        import cv2
    except ImportError:
        print("Requires: pyrealsense2, opencv-python, numpy")
        sys.exit(1)

    # Load calibration if it exists
    cal = None
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            cal = yaml.safe_load(f).get("zones")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_scale = (pipeline.get_active_profile()
                   .get_device().first_depth_sensor()
                   .get_depth_scale())

    print("Showing depth preview. Press Q to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_m = depth_image * depth_scale

            # Colorize depth
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Overlay zone highlighting
            if cal:
                panel = cal.get("panel", {})
                pz_min = panel.get("z_min", 0.3)
                pz_max = panel.get("z_max", 0.7)

                mask_panel = np.logical_and(
                    depth_m > pz_min, depth_m < pz_max
                ).astype(np.uint8) * 255

                panel_overlay = np.zeros_like(color_image)
                panel_overlay[:, :, 2] = mask_panel   # Red channel

                depth_vis = cv2.addWeighted(depth_vis, 0.7, panel_overlay, 0.3, 0)

            # Info text
            h, w = depth_image.shape
            cx, cy = w // 2, h // 2
            center_depth = depth_m[cy, cx]
            cv2.putText(depth_vis, f"Center depth: {center_depth:.3f}m",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            if cal:
                cv2.putText(depth_vis, f"Panel zone: {cal['panel']['z_min']:.2f} - {cal['panel']['z_max']:.2f}m",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
                cv2.putText(depth_vis, "(red = panel zone)", (10, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,255), 1)

            cv2.imshow("Lavender Zone Preview", depth_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()
    else:
        calibrate()
