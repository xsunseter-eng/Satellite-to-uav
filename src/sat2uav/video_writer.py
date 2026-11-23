"""FFmpeg H.264 MP4 video writer and structured telemetry/metadata exporter."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class FFmpegVideoWriter:
    """Streams uint8 RGB24 frames directly to ffmpeg to produce H.264 MP4 videos."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: int,
        crf: int = 20,
    ) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg binary not found on PATH.")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # H.264 yuv420p requires even width and height
        self.width = int(width) & ~1
        self.height = int(height) & ~1
        self.fps = int(fps)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def write_frame(self, rgb_frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("FFmpeg stdin is closed.")
        frame = rgb_frame[: self.height, : self.width, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr_bytes = self._proc.stderr.read() if self._proc.stderr else b""
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg failed (code {ret}): {stderr_bytes.decode('utf-8', errors='replace')}"
            )

    def __enter__(self) -> "FFmpegVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_video_metadata(output_json: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
