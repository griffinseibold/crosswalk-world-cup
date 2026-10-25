"""Decode recorded footage with PyAV.

PyAV was chosen over OpenCV because its wheels bundle a known FFmpeg build (so
pinning ``av`` pins the decoder), it exposes exact presentation timestamps for
each frame (needed to pace playback like a real camera and to label
detections with a time), and it can resize and convert colour inside FFmpeg's
swscale while decoding instead of in Python.

Only recorded files are supported. There is deliberately no RTSP or webcam
input: see docs/data.md for why the project does not process live feeds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts"}


@dataclass
class Frame:
    video: str
    index: int
    pts_s: float
    image: np.ndarray  # H x W x 3, uint8, RGB

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class VideoInfo:
    path: str
    fps: float
    frames: int | None
    width: int
    height: int
    codec: str
    duration_s: float | None


def list_videos(paths: list[str]) -> list[str]:
    """Expand local files and directories into a sorted list of video files."""
    found: list[str] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(
                str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in VIDEO_SUFFIXES
            )
        elif path.is_file():
            found.append(str(path))
        else:
            raise FileNotFoundError(item)
    return found


def probe(path: str) -> VideoInfo:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or 30
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        return VideoInfo(
            path=path,
            fps=float(rate),
            frames=stream.frames or None,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=stream.codec_context.name,
            duration_s=duration,
        )


def iter_frames(
    path: str,
    resize: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        # Let FFmpeg decode on several threads; the GPU, not the decoder,
        # should be the bottleneck being measured.
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        rate = float(stream.average_rate or 30)
        for index, frame in enumerate(container.decode(stream)):
            if max_frames is not None and index >= max_frames:
                break
            if resize is not None:
                frame = frame.reformat(width=resize[0], height=resize[1], format="rgb24")
                image = frame.to_ndarray()
            else:
                image = frame.to_ndarray(format="rgb24")
            pts_s = frame.pts * time_base if frame.pts is not None else index / rate
            yield Frame(video=path, index=index, pts_s=float(pts_s), image=image)


def write_test_video(path: str, frames: int = 30, size: tuple[int, int] = (320, 240), fps: int = 30):
    """Encode a short synthetic clip (moving box) for tests and smoke runs."""
    import av

    width, height = size
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 5) % max(1, width - 40)
            image[height // 3 : height // 3 + 40, x : x + 40] = (255, 200, 0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
