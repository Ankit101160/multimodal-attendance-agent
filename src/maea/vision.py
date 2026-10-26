
"""Vision module: face detection, tracking, head-pose proxy and motion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config


# --------------------------------------------------
# Face detector initialization
# --------------------------------------------------

# Project root: multimodal-attendance-agent/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTAL_XML = PROJECT_ROOT / "haarcascade_frontalface_default.xml"
PROFILE_XML = PROJECT_ROOT / "haarcascade_profileface.xml"


def load_cascade(xml_path: Path) -> cv2.CascadeClassifier:
    """Load a Haar Cascade XML file and validate the classifier."""
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Face detector XML file not found: {xml_path}\n"
            "Please place the required XML file in the project root."
        )

    classifier = cv2.CascadeClassifier(str(xml_path))

    if classifier.empty():
        raise RuntimeError(
            f"OpenCV could not load the classifier: {xml_path}"
        )

    return classifier


_FRONTAL = load_cascade(FRONTAL_XML)
_PROFILE = load_cascade(PROFILE_XML)

_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# --------------------------------------------------
# Face detection data model
# --------------------------------------------------

@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    frontal: bool

    @property
    def center(self):
        return (
            self.x + self.w / 2.0,
            self.y + self.h / 2.0
        )


def _iou(a: Detection, b: Detection) -> float:
    """Calculate intersection over union of two face boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter

    return inter / union if union else 0.0


def detect_faces(
    gray: np.ndarray,
    min_px: int = 40
) -> list[Detection]:
    """Detect frontal and profile faces in a grayscale frame."""

    if gray is None or gray.size == 0:
        return []

    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = _CLAHE.apply(gray)

    out: list[Detection] = []

    # Frontal faces
    frontal_faces = _FRONTAL.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_px, min_px)
    )

    for x, y, w, h in frontal_faces:
        out.append(
            Detection(
                int(x), int(y), int(w), int(h), True
            )
        )

    # Profile faces
    width = gray.shape[1]

    profiles = list(
        _PROFILE.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(min_px, min_px)
        )
    )

    # Detect faces looking in the opposite direction
    flipped = cv2.flip(gray, 1)

    flipped_profiles = _PROFILE.detectMultiScale(
        flipped,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_px, min_px)
    )

    for x, y, w, h in flipped_profiles:
        profiles.append((width - x - w, y, w, h))

    for x, y, w, h in profiles:
        detection = Detection(
            int(x), int(y), int(w), int(h), False
        )

        # Avoid duplicate detections of frontal faces
        if not any(_iou(detection, face) > 0.3 for face in out):
            out.append(detection)

    return out


# --------------------------------------------------
# Tracking data model
# --------------------------------------------------

@dataclass
class Track:
    tid: int
    center: tuple
    width: float
    last_t: float


@dataclass
class CentroidTracker:
    cfg: Config
    tracks: dict = field(default_factory=dict)
    retired: dict = field(default_factory=dict)
    _next: int = 1

    def update(
        self,
        dets: list[Detection],
        t: float
    ) -> list[tuple[int, Detection]]:

        assigned: list[tuple[int, Detection]] = []
        free = set(self.tracks)

        for det in sorted(dets, key=lambda d: -d.w * d.h):
            best = None
            best_d = None

            for tid in free:
                tr = self.tracks[tid]

                dist = np.hypot(
                    det.center[0] - tr.center[0],
                    det.center[1] - tr.center[1]
                ) / max(tr.width, 1)

                if (
                    dist <= self.cfg.max_match_dist
                    and (best_d is None or dist < best_d)
                ):
                    best = tid
                    best_d = dist

            if best is None:
                best = self._reclaim(det)

            if best is None:
                best = self._next
                self._next += 1

                self.tracks[best] = Track(
                    best, det.center, float(det.w), t
                )

            elif best not in self.tracks:
                self.tracks[best] = Track(
                    best, det.center, float(det.w), t
                )

            else:
                free.discard(best)

            tr = self.tracks[best]
            tr.center = det.center
            tr.width = float(det.w)
            tr.last_t = t

            assigned.append((best, det))

        # Retire tracks missing for too long
        for tid in list(self.tracks):
            if (
                t - self.tracks[tid].last_t
                > self.cfg.max_missing_sec
            ):
                self.retired[tid] = self.tracks.pop(tid)

        return assigned

    def _reclaim(self, det: Detection):
        """Re-identify a returning person by approximate seat position."""
        best = None
        best_d = None

        for tid, tr in self.retired.items():
            dist = np.hypot(
                det.center[0] - tr.center[0],
                det.center[1] - tr.center[1]
            ) / max(tr.width, 1)

            if dist <= 0.8 and (best_d is None or dist < best_d):
                best = tid
                best_d = dist

        if best is not None:
            del self.retired[best]

        return best


# --------------------------------------------------
# Video analysis
# --------------------------------------------------

def analyse_video(
    path: str,
    cfg: Config,
    annotated_out: str | None = None,
    progress=None
):
    """Return records and metadata for sampled video frames."""

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / cfg.sample_fps)))

    tracker = CentroidTracker(cfg)
    prev_center: dict[int, tuple] = {}
    records: list[dict] = []

    writer = None
    idx = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if idx % step == 0:
                t = idx / fps

                if frame is None or frame.size == 0:
                    idx += 1
                    continue

                scale = cfg.resize_width / frame.shape[1]

                small = (
                    cv2.resize(frame, None, fx=scale, fy=scale)
                    if scale < 1
                    else frame
                )

                gray = cv2.cvtColor(
                    small, cv2.COLOR_BGR2GRAY
                )

                assigned = tracker.update(
                    detect_faces(gray, cfg.min_face_px),
                    t
                )

                for tid, det in assigned:
                    px, py = prev_center.get(tid, det.center)

                    motion = float(
                        np.hypot(
                            det.center[0] - px,
                            det.center[1] - py
                        ) / max(det.w, 1)
                    )

                    prev_center[tid] = det.center

                    records.append({
                        "t": round(t, 3),
                        "pid": tid,
                        "frontal": int(det.frontal),
                        "motion": motion
                    })

                # Optional annotated output video
                if annotated_out:
                    vis = small.copy()

                    for tid, det in assigned:
                        color = (
                            (0, 200, 0)
                            if det.frontal
                            else (0, 165, 255)
                        )

                        cv2.rectangle(
                            vis,
                            (det.x, det.y),
                            (det.x + det.w, det.y + det.h),
                            color,
                            2
                        )

                        label = (
                            f"P{tid} "
                            f"{'attentive' if det.frontal else 'looking away'}"
                        )

                        cv2.putText(
                            vis,
                            label,
                            (det.x, max(15, det.y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                    if writer is None:
                        h, w = vis.shape[:2]
                        output_path = Path(annotated_out)
                        output_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )

                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            cfg.sample_fps,
                            (w, h)
                        )

                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError(
                                f"Could not create output video: {output_path}"
                            )

                    writer.write(vis)

            idx += 1

            if progress and total and idx % 30 == 0:
                progress(min(idx / total, 1.0))

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    if idx == 0:
        raise ValueError(f"Video has no readable frames: {path}")

    if progress:
        progress(1.0)

    return records, {
        "fps": fps,
        "duration": idx / fps if fps else 0.0,
        "frames": idx
    }
