"""
Minimalist, high-performance UI rendering and prediction smoothing for OpenCV.

Focuses exclusively on essential information:
- Clean corner guide brackets (no center crosshairs or clutter).
- Centered floating bottom pill displaying denomination, confidence, and status.
- Zero Unicode encoding artifacts (clean ASCII text).
- Toast notification for snapshots.
"""

import time
from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

import config


class PredictionSmoother:
    """
    Manages rolling prediction history to eliminate transient noise
    and jitter while keeping response times fast and responsive.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_consistency_count: int = 2,
        confidence_threshold: float = 0.40,
    ):
        self.window_size = window_size
        self.min_consistency_count = min_consistency_count
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=window_size)
        self.current_stable_class: Optional[str] = None
        self.current_confidence: float = 0.0
        self.status_state: str = "NO_CURRENCY"

    def update(self, predicted_class: str, confidence: float) -> Tuple[Optional[str], float, str]:
        """
        Update rolling prediction consensus.
        Returns:
            (stable_class, smoothed_confidence, status_state)
        """
        self.history.append((predicted_class, float(confidence)))

        if len(self.history) < 2:
            self.current_stable_class = predicted_class if predicted_class != "unknown" else None
            self.current_confidence = confidence
            self.status_state = "CURRENCY_DETECTED" if predicted_class != "unknown" else "NO_CURRENCY"
            return self.current_stable_class, self.current_confidence, self.status_state

        class_counts = Counter(item[0] for item in self.history)
        most_common_class, top_count = class_counts.most_common(1)[0]

        matching_confidences = [item[1] for item in self.history if item[0] == most_common_class]
        avg_confidence = float(np.mean(matching_confidences)) if matching_confidences else 0.0
        self.current_confidence = avg_confidence

        if most_common_class == "unknown":
            self.current_stable_class = "unknown"
            self.status_state = "NO_CURRENCY"
            return self.current_stable_class, self.current_confidence, self.status_state

        if top_count >= self.min_consistency_count:
            self.current_stable_class = most_common_class
            self.status_state = "CURRENCY_DETECTED"
        else:
            self.current_stable_class = most_common_class
            self.status_state = "UNCERTAIN"

        return self.current_stable_class, self.current_confidence, self.status_state

    def reset(self):
        """Clear smoothing history."""
        self.history.clear()
        self.current_stable_class = None
        self.current_confidence = 0.0
        self.status_state = "NO_CURRENCY"


def get_roi_coordinates(frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """Calculate centered bounding box coordinates (x1, y1, x2, y2) for the note ROI."""
    roi_w = min(720, int(frame_width * 0.72))
    roi_h = min(340, int(frame_height * 0.62))

    x1 = (frame_width - roi_w) // 2
    y1 = (frame_height - roi_h) // 2 - 10
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)


def draw_rounded_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 12,
    thickness: int = -1,
):
    """Draw an anti-aliased rounded rectangle."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return

    radius = min(radius, w // 2, h // 2)

    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)


def draw_roi_guide(
    frame: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    accent_color: Tuple[int, int, int] = (240, 200, 70),
    corner_len: int = 35,
    thickness: int = 3,
):
    """Draw sleek modern corner brackets indicating where to hold the banknote."""
    x1, y1, x2, y2 = roi_coords

    # Subtle boundary line
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # Top-Left Corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), accent_color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), accent_color, thickness)

    # Top-Right Corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), accent_color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), accent_color, thickness)

    # Bottom-Left Corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), accent_color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), accent_color, thickness)

    # Bottom-Right Corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), accent_color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), accent_color, thickness)


def draw_minimal_hud(
    frame: np.ndarray,
    stable_class: Optional[str],
    confidence: float,
    status_state: str,
    fps: float,
    toast_message: Optional[str] = None,
) -> np.ndarray:
    """
    Renders an ultra-clean, distraction-free interface:
    - NO giant header bars.
    - NO crosshairs on the user's face.
    - Centered floating bottom pill with Denomination, Confidence, and Status.
    - Small discreet keyboard controls in top-right corner.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # 1. DISCREET TOP-RIGHT CONTROLS PILL
    pill_w = 270
    pill_h = 32
    rx2 = w - 24
    rx1 = rx2 - pill_w
    ry1 = 18
    ry2 = ry1 + pill_h

    ctrl_overlay = canvas.copy()
    draw_rounded_rectangle(ctrl_overlay, (rx1, ry1), (rx2, ry2), (20, 22, 28), radius=8)
    cv2.addWeighted(ctrl_overlay, 0.75, canvas, 0.25, 0, canvas)
    draw_rounded_rectangle(canvas, (rx1, ry1), (rx2, ry2), (50, 56, 68), radius=8, thickness=1)

    controls_text = f"[Q] Quit  |  [S] Save  |  FPS {fps:3.0f}"
    cv2.putText(
        canvas,
        controls_text,
        (rx1 + 14, ry1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 205, 215),
        1,
        cv2.LINE_AA,
    )

    # 2. CENTERED FLOATING BOTTOM CARD
    card_w = min(620, w - 80)
    card_h = 96
    cx1 = (w - card_w) // 2
    cx2 = cx1 + card_w
    cy1 = h - card_h - 28
    cy2 = cy1 + card_h

    card_overlay = canvas.copy()
    draw_rounded_rectangle(card_overlay, (cx1, cy1), (cx2, cy2), (18, 20, 26), radius=16)
    cv2.addWeighted(card_overlay, 0.90, canvas, 0.10, 0, canvas)

    # Determine status details
    if status_state == "CURRENCY_DETECTED" and stable_class in config.CLASS_METADATA:
        meta = config.CLASS_METADATA[stable_class]
        main_text = meta["display_name"]
        color_accent = meta["color_bgr"]
        badge_text = "CURRENCY DETECTED"
        badge_bg = (50, 190, 80)  # Green
        badge_fg = (10, 10, 10)
        card_border_color = color_accent
    elif status_state == "UNCERTAIN":
        main_text = "HOLD NOTE STEADY"
        color_accent = (50, 180, 245)  # Amber
        badge_text = "ALIGNING NOTE..."
        badge_bg = (50, 180, 245)
        badge_fg = (15, 15, 15)
        card_border_color = (60, 68, 78)
    else:
        main_text = "NO CURRENCY DETECTED"
        color_accent = (150, 155, 165)
        badge_text = "SCANNING"
        badge_bg = (50, 56, 68)
        badge_fg = (210, 215, 220)
        card_border_color = (50, 56, 68)

    draw_rounded_rectangle(canvas, (cx1, cy1), (cx2, cy2), card_border_color, radius=16, thickness=2)

    # Status Badge Pill (Top-Left of Card)
    bw = 180
    bh = 22
    bx1 = cx1 + 22
    by1 = cy1 + 16
    draw_rounded_rectangle(canvas, (bx1, by1), (bx1 + bw, by1 + bh), badge_bg, radius=5)
    cv2.putText(
        canvas,
        badge_text,
        (bx1 + 10, by1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        badge_fg,
        1,
        cv2.LINE_AA,
    )

    # Main Denomination Title
    cv2.putText(
        canvas,
        main_text,
        (cx1 + 22, cy1 + 68),
        cv2.FONT_HERSHEY_DUPLEX,
        0.82,
        color_accent,
        2,
        cv2.LINE_AA,
    )

    # Confidence Indicator (Right of Card)
    conf_pct = max(0.0, min(100.0, confidence * 100.0))
    conf_str = f"Confidence: {conf_pct:4.1f}%" if status_state == "CURRENCY_DETECTED" else "Awaiting note"
    (tw, _), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(
        canvas,
        conf_str,
        (cx2 - tw - 24, cy1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 235, 240),
        1,
        cv2.LINE_AA,
    )

    # Mini Confidence Bar
    if status_state == "CURRENCY_DETECTED":
        bar_w = 170
        bar_h = 7
        bx = cx2 - bar_w - 24
        by = cy1 + 44
        draw_rounded_rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (45, 50, 60), radius=3)
        fill_w = int(bar_w * (conf_pct / 100.0))
        if fill_w > 4:
            draw_rounded_rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h), color_accent, radius=3)

    # 3. TOAST NOTIFICATION (Saved Snapshot)
    if toast_message:
        tw, _ = cv2.getTextSize(toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        tx = (w - tw) // 2
        ty = 30
        draw_rounded_rectangle(canvas, (tx - 18, ty - 20), (tx + tw + 18, ty + 16), (30, 140, 60), radius=8)
        cv2.putText(
            canvas,
            toast_message,
            (tx, ty + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas
