"""CLI utility to generate synthetic drone test footage and telemetry for demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def generate_demo_dataset(output_dir: str = "data/demo", duration_s: int = 4, fps: int = 15):
    """Generate high-texture synthetic aerial flight video and GPS telemetry."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    video_path = out_p / "demo_flight.mp4"
    json_path = out_p / "demo_telemetry.json"
    csv_path = out_p / "demo_telemetry.csv"

    width, height = 640, 480
    n_frames = duration_s * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Base synthetic ground landscape
    np.random.seed(99)
    canvas_w, canvas_h = width + 400, height + 400
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (34, 139, 34)  # Forest green ground

    # Add geometric features: buildings, roads, water pond
    cv2.rectangle(canvas, (0, canvas_h // 2 - 20), (canvas_w, canvas_h // 2 + 20), (80, 80, 80), -1)  # Road
    cv2.ellipse(canvas, (canvas_w // 2, canvas_h // 2 + 100), (90, 60), 0, 0, 360, (180, 100, 40), -1)  # Pond

    # Add structures
    for r in range(40, canvas_h - 40, 60):
        for c in range(40, canvas_w - 40, 70):
            b_color = (int(np.random.randint(120, 220)), int(np.random.randint(120, 220)), int(np.random.randint(120, 220)))
            cv2.rectangle(canvas, (c, r), (c + 45, r + 45), b_color, -1)
            cv2.rectangle(canvas, (c, r), (c + 45, r + 45), (30, 30, 30), 2)
            cv2.circle(canvas, (c + 22, r + 22), 8, (240, 240, 240), -1)

    lat0, lon0, alt0 = 28.6139, 77.2090, 120.0
    telemetry_json = []
    csv_lines = ["timestamp_s,latitude,longitude,altitude_m,pitch_deg,roll_deg,yaw_deg\n"]

    for i in range(n_frames):
        t_sec = i / fps
        y_off = int(i * 3.5)
        x_off = int(i * 2.0)
        frame = canvas[y_off : y_off + height, x_off : x_off + width].copy()

        # Add simulated timestamp overlay
        cv2.putText(
            frame,
            f"AERORECON DRONE REC - {t_sec:05.2f}s",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

        lat = lat0 + (i * 0.000015)
        lon = lon0 + (i * 0.000025)
        alt = alt0 + (i * 0.1)

        telemetry_json.append({
            "timestamp_s": round(t_sec, 2),
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "altitude_m": round(alt, 2),
            "pitch_deg": -45.0,
            "roll_deg": 0.0,
            "yaw_deg": round(float(i * 0.8), 2),
        })

        csv_lines.append(
            f"{t_sec:.2f},{lat:.7f},{lon:.7f},{alt:.2f},-45.0,0.0,{float(i * 0.8):.2f}\n"
        )

    writer.release()
    json_path.write_text(json.dumps(telemetry_json, indent=2))
    csv_path.write_text("".join(csv_lines))

    print(f"Generated demo dataset:")
    print(f"  Video:     {video_path} ({video_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Telemetry: {json_path}")
    print(f"             {csv_path}")


if __name__ == "__main__":
    generate_demo_dataset()
