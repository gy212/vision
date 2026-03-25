from __future__ import annotations

from pathlib import Path

import cv2


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def open_video_writer(preferred_path: Path, *, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, str]:
    """
    Open a VideoWriter with best-effort codec/container compatibility.

    Returns: (writer, actual_path, codec_tag)
    """
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid frame size: {size}")

    fps = float(fps) if fps and fps > 1e-6 else 30.0
    p = Path(preferred_path)
    suf = p.suffix.lower()

    candidates: list[tuple[Path, str]] = []
    if suf == ".mp4":
        # H.264 is widely supported, but may be unavailable in some OpenCV builds.
        # If H.264 isn't available, prefer AVI+MJPG over MP4V for better player compatibility.
        candidates = [(p, "avc1"), (p, "H264"), (p.with_suffix(".avi"), "MJPG"), (p, "mp4v")]
    elif suf == ".avi":
        candidates = [(p, "MJPG"), (p, "XVID")]
    else:
        # Unknown extension: keep user's path first, then fall back to AVI.
        candidates = [(p, "mp4v"), (p.with_suffix(".avi"), "MJPG")]

    last_err: str | None = None
    for out_path, codec in candidates:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_path), _fourcc(codec), fps, (w, h))
        if vw.isOpened():
            return vw, out_path, codec
        last_err = f"VideoWriter open failed: path={out_path}, codec={codec}"

    raise RuntimeError(last_err or "VideoWriter open failed")
