"""
UI rendering and prediction smoothing helper functions for OpenCV interface.

Provides:
- Thread-safe prediction smoothing with rolling window and temporal consensus.
- Image quality checks (Laplacian blur metric, exposure checks).
- Glassmorphic dark HUD drawing with status badges, confidence meters, and guide reticles.
"""

import time
from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

import config


class PredictionSmoother:
    """
    Manages a rolling window of recent frame predictions to stabilize
    denomination outputs and reject rapid jitter or transient misclassifications.
    """

    def __init__(
        self,
        window_size: int = config.SMOOTHING_WINDOW_SIZE,
        min_consistency_count: int = config.STABILITY_MIN_COUNT,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        unknown_threshold: float = config.UNKNOWN_CONFIDENCE_THRESHOLD,
    ):
        self.window_size = window_size
        self.min_consistency_count = min_consistency_count
        self.confidence_threshold = confidence_threshold
        self.unknown_threshold = unknown_threshold
        self.history = deque(maxlen=window_size)
        self.current_stable_class: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_stability_ratio: float = 0.0
        self.status_state: str = "NO_INPUT"

    def update(self, predicted_class: str, confidence: float) -> Tuple[Optional[str], float, float, str]:
        """
        Add a new prediction to the rolling history and compute the stable consensus.

        Returns:
            (stable_class, smoothed_confidence, stability_ratio, status_state)
            status_state can be:
                - "CURRENCY_DETECTED"
                - "NO_CURRENCY"
                - "LOW_CONFIDENCE"
                - "COLLECTING_EVIDENCE"
        """
        self.history.append((predicted_class, float(confidence)))

        if len(self.history) < 3:
            self.current_stable_class = None
            self.current_confidence = confidence
            self.current_stability_ratio = len(self.history) / self.window_size
            self.status_state = "COLLECTING_EVIDENCE"
            return self.current_stable_class, self.current_confidence, self.current_stability_ratio, self.status_state

        # Count frequencies of each predicted class in recent window
        class_counts = Counter(item[0] for item in self.history)
        most_common_class, top_count = class_counts.most_common(1)[0]

        # Calculate average confidence for the dominant class
        matching_confidences = [item[1] for item in self.history if item[0] == most_common_class]
        avg_confidence = float(np.mean(matching_confidences)) if matching_confidences else 0.0
        stability_ratio = top_count / len(self.history)

        self.current_confidence = avg_confidence
        self.current_stability_ratio = stability_ratio

        # Case 1: Unknown / No Currency class dominant
        if most_common_class == "unknown":
            if top_count >= self.min_consistency_count and avg_confidence >= self.unknown_threshold:
                self.current_stable_class = "unknown"
                self.status_state = "NO_CURRENCY"
            else:
                self.current_stable_class = None
                self.status_state = "LOW_CONFIDENCE"
            return self.current_stable_class, self.current_confidence, self.current_stability_ratio, self.status_state

        # Case 2: Banknote denomination candidate
        if top_count >= self.min_consistency_count:
            if avg_confidence >= self.confidence_threshold:
                self.current_stable_class = most_common_class
                self.status_state = "CURRENCY_DETECTED"
            else:
                self.current_stable_class = None
                self.status_state = "LOW_CONFIDENCE"
        else:
            self.current_stable_class = None
            self.status_state = "COLLECTING_EVIDENCE"

        return self.current_stable_class, self.current_confidence, self.current_stability_ratio, self.status_state

    def reset(self):
        """Reset the smoothing buffer completely."""
        self.history.clear()
        self.current_stable_class = None
        self.current_confidence = 0.0
        self.current_stability_ratio = 0.0
        self.status_state = "NO_INPUT"


def check_image_quality(roi_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate visual quality of the Region of Interest:
    - Laplacian variance for blur detection.
    - Mean grayscale brightness for underexposure / overexposure.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return {
            "is_valid": False,
            "blur_metric": 0.0,
            "brightness": 0.0,
            "is_blurry": True,
            "is_dark": False,
            "is_overexposed": False,
            "warning": "Empty Frame",
        }

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    is_blurry = blur_metric < config.BLUR_THRESHOLD
    is_dark = brightness < config.MIN_BRIGHTNESS
    is_overexposed = brightness > config.MAX_BRIGHTNESS

    warning = None
    if is_blurry:
        warning = "BLUR DETECTED - HOLD STEADY"
    elif is_dark:
        warning = "POOR LIGHTING - INCREASE LIGHT"
    elif is_overexposed:
        warning = "GLARE / OVEREXPOSURE DETECTED"

    return {
        "is_valid": True,
        "blur_metric": blur_metric,
        "brightness": brightness,
        "is_blurry": is_blurry,
        "is_dark": is_dark,
        "is_overexposed": is_overexposed,
        "warning": warning,
    }


def get_roi_coordinates(frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """Calculate centered bounding box coordinates (x1, y1, x2, y2) for the note ROI."""
    roi_w = min(config.ROI_WIDTH, int(frame_width * 0.75))
    roi_h = min(config.ROI_HEIGHT, int(frame_height * 0.65))

    # Keep a standard 2.1:1 ratio
    x1 = (frame_width - roi_w) // 2
    y1 = (frame_height - roi_h) // 2 - 20  # slightly shifted up to allow lower HUD
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # Boundary protection
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    return x1, y1, x2, y2


def draw_rounded_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
):
    """Draw a smooth rounded rectangle on an image."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return

    radius = min(radius, w // 2, h // 2)

    if thickness == -1:
        # Filled
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Border outline
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
    accent_color: Tuple[int, int, int] = config.UI_THEME["roi_guide"],
    corner_len: int = 30,
    thickness: int = 3,
):
    """Draw corner-accented guide reticle indicating where user should position the note."""
    x1, y1, x2, y2 = roi_coords

    # Thin semi-transparent border
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

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

    # Subtle central crosshair
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (200, 200, 200), 1)

    # Helper label above ROI
    label = "ALIGN BANKNOTE INSIDE GUIDE FRAME"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        frame,
        label,
        (cx - tw // 2, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 230),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    stable_class: Optional[str],
    confidence: float,
    stability_ratio: float,
    status_state: str,
    fps: float,
    quality_info: Dict[str, Any],
    toast_message: Optional[str] = None,
    show_help: bool = False,
) -> np.ndarray:
    """
    Draw a complete dark, clean, professional HUD interface over the frame.
    All text elements have high-contrast background backing for readability
    under any camera lighting condition.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # 1. TOP HEADER BAR
    header_h = 56
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), config.UI_THEME["bg_dark"], -1)
    cv2.line(overlay, (0, header_h), (w, header_h), config.UI_THEME["panel_border"], 1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

    # Title & Subtitle
    cv2.circle(canvas, (24, 28), 6, config.UI_THEME["accent_cyan"], -1)
    cv2.circle(canvas, (24, 28), 8, config.UI_THEME["accent_cyan"], 1)
    cv2.putText(
        canvas,
        "INDIAN CURRENCY NOTE CLASSIFIER",
        (40, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        config.UI_THEME["text_primary"],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Deep Learning Real-Time Webcam Classifier  |  MobileNetV2",
        (42, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        config.UI_THEME["text_secondary"],
        1,
        cv2.LINE_AA,
    )

    # Top-right Live FPS & Status Indicator
    fps_text = f"FPS: {fps:4.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        canvas,
        fps_text,
        (w - tw - 24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        config.UI_THEME["text_secondary"],
        1,
        cv2.LINE_AA,
    )

    # 2. BOTTOM FLOATING PREDICTION PANEL
    panel_h = 136
    panel_y1 = h - panel_h - 20
    panel_y2 = h - 20
    panel_x1 = 24
    panel_x2 = w - 24

    panel_overlay = canvas.copy()
    draw_rounded_rectangle(
        panel_overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        config.UI_THEME["panel_bg"],
        radius=14,
        thickness=-1,
    )
    cv2.addWeighted(panel_overlay, 0.90, canvas, 0.10, 0, canvas)
    draw_rounded_rectangle(
        canvas,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        config.UI_THEME["panel_border"],
        radius=14,
        thickness=1,
    )

    # Determine status pill styling & text
    warning_text = quality_info.get("warning")
    if warning_text and status_state not in ("CURRENCY_DETECTED", "NO_CURRENCY"):
        pill_text = warning_text
        pill_bg = config.UI_THEME["accent_amber"]
        pill_fg = (15, 15, 15)
    elif status_state == "CURRENCY_DETECTED" and stable_class:
        pill_text = "CURRENCY DETECTED"
        pill_bg = config.UI_THEME["accent_green"]
        pill_fg = (10, 10, 10)
    elif status_state == "NO_CURRENCY":
        pill_text = "NO CURRENCY DETECTED"
        pill_bg = config.UI_THEME["panel_border"]
        pill_fg = config.UI_THEME["text_primary"]
    elif status_state == "LOW_CONFIDENCE":
        pill_text = "LOW CONFIDENCE - HOLD STEADY"
        pill_bg = config.UI_THEME["accent_amber"]
        pill_fg = (15, 15, 15)
    else:
        pill_text = "SCANNING FOR CURRENCY..."
        pill_bg = (55, 60, 70)
        pill_fg = config.UI_THEME["text_secondary"]

    # Draw Status Pill Badge
    pill_w = 260
    pill_h = 24
    pill_x = panel_x1 + 20
    pill_y = panel_y1 + 16
    draw_rounded_rectangle(
        canvas,
        (pill_x, pill_y),
        (pill_x + pill_w, pill_y + pill_h),
        pill_bg,
        radius=6,
        thickness=-1,
    )
    cv2.putText(
        canvas,
        pill_text,
        (pill_x + 10, pill_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        pill_fg,
        1,
        cv2.LINE_AA,
    )

    # Display Main Predicted Result Title
    result_x = panel_x1 + 20
    result_y = panel_y1 + 72

    if status_state == "CURRENCY_DETECTED" and stable_class in config.CLASS_METADATA:
        meta = config.CLASS_METADATA[stable_class]
        main_text = meta["display_name"]
        color_accent = meta["color_bgr"]
        desc_text = meta["description"]
    elif status_state == "NO_CURRENCY":
        main_text = "NO CURRENCY / UNKNOWN OBJECT"
        color_accent = config.CLASS_METADATA["unknown"]["color_bgr"]
        desc_text = "Rejection Class Active (No valid Indian note detected)"
    elif status_state == "LOW_CONFIDENCE":
        main_text = "UNCERTAIN — HOLD NOTE STEADY"
        color_accent = config.UI_THEME["accent_amber"]
        desc_text = "Confidence below threshold or rapid movement detected"
    else:
        main_text = "PRESENT INDIAN BANKNOTE IN FRAME"
        color_accent = config.UI_THEME["text_muted"]
        desc_text = "Place note inside guide frame for real-time classification"

    # Main denomination text
    cv2.putText(
        canvas,
        main_text,
        (result_x, result_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        color_accent,
        2,
        cv2.LINE_AA,
    )

    # Description subtitle
    cv2.putText(
        canvas,
        desc_text,
        (result_x, result_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        config.UI_THEME["text_secondary"],
        1,
        cv2.LINE_AA,
    )

    # RIGHT SIDE OF PANEL: Confidence & Stability Progress Bars
    meter_x1 = max(result_x + 480, panel_x2 - 380)
    meter_y1 = panel_y1 + 20
    meter_w = panel_x2 - meter_x1 - 24

    # 1. Confidence Meter
    conf_pct = max(0.0, min(100.0, confidence * 100.0))
    conf_label = f"Confidence: {conf_pct:5.1f}%"
    cv2.putText(
        canvas,
        conf_label,
        (meter_x1, meter_y1 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        config.UI_THEME["text_primary"],
        1,
        cv2.LINE_AA,
    )

    # Confidence bar track & fill
    bar_h = 10
    bar_y = meter_y1 + 22
    draw_rounded_rectangle(
        canvas,
        (meter_x1, bar_y),
        (meter_x1 + meter_w, bar_y + bar_h),
        (45, 50, 60),
        radius=4,
        thickness=-1,
    )

    fill_w = int(meter_w * (conf_pct / 100.0))
    if fill_w > 4:
        fill_color = (
            config.UI_THEME["accent_green"]
            if conf_pct >= 70
            else (config.UI_THEME["accent_amber"] if conf_pct >= 50 else config.UI_THEME["accent_red"])
        )
        draw_rounded_rectangle(
            canvas,
            (meter_x1, bar_y),
            (meter_x1 + fill_w, bar_y + bar_h),
            fill_color,
            radius=4,
            thickness=-1,
        )

    # 2. Stability Consensus Meter
    stab_pct = max(0.0, min(100.0, stability_ratio * 100.0))
    stab_y = meter_y1 + 54
    stab_label = f"Stability:  {stab_pct:5.1f}% ({int(stability_ratio * config.SMOOTHING_WINDOW_SIZE)}/{config.SMOOTHING_WINDOW_SIZE} frames)"
    cv2.putText(
        canvas,
        stab_label,
        (meter_x1, stab_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        config.UI_THEME["text_secondary"],
        1,
        cv2.LINE_AA,
    )

    stab_bar_y = stab_y + 22
    draw_rounded_rectangle(
        canvas,
        (meter_x1, stab_bar_y),
        (meter_x1 + meter_w, stab_bar_y + bar_h),
        (45, 50, 60),
        radius=4,
        thickness=-1,
    )
    stab_fill_w = int(meter_w * (stab_pct / 100.0))
    if stab_fill_w > 4:
        draw_rounded_rectangle(
            canvas,
            (meter_x1, stab_bar_y),
            (meter_x1 + stab_fill_w, stab_bar_y + bar_h),
            config.UI_THEME["accent_cyan"],
            radius=4,
            thickness=-1,
        )

    # Small Keyboard Shortcuts Footer inside panel
    shortcuts_str = "[Q] Quit  |  [S] Save Snapshot  |  [R] Reset Smoothing  |  [H] Toggle Help"
    cv2.putText(
        canvas,
        shortcuts_str,
        (panel_x1 + 20, panel_y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        config.UI_THEME["text_muted"],
        1,
        cv2.LINE_AA,
    )

    # 3. TOAST NOTIFICATION BANNER (if active)
    if toast_message:
        toast_w = min(680, w - 80)
        toast_h = 36
        toast_x = (w - toast_w) // 2
        toast_y = header_h + 16
        draw_rounded_rectangle(
            canvas,
            (toast_x, toast_y),
            (toast_x + toast_w, toast_y + toast_h),
            (25, 75, 30),
            radius=8,
            thickness=-1,
        )
        draw_rounded_rectangle(
            canvas,
            (toast_x, toast_y),
            (toast_x + toast_w, toast_y + toast_h),
            config.UI_THEME["accent_green"],
            radius=8,
            thickness=1,
        )
        (tw, _), _ = cv2.getTextSize(toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas,
            toast_message,
            (toast_x + (toast_w - tw) // 2, toast_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 4. HELP OVERLAY DRAWER (if toggled)
    if show_help:
        draw_help_modal(canvas, w, h)

    return canvas


def draw_help_modal(canvas: np.ndarray, width: int, height: int):
    """Draw interactive help overlay showing instructions and keyboard shortcuts."""
    modal_w = min(560, width - 40)
    modal_h = min(400, height - 80)
    x1 = (width - modal_w) // 2
    y1 = (height - modal_h) // 2
    x2 = x1 + modal_w
    y2 = y1 + modal_h

    # Backdrop blur / dim
    dim_overlay = canvas.copy()
    cv2.rectangle(dim_overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(dim_overlay, 0.55, canvas, 0.45, 0, canvas)

    # Modal window card
    draw_rounded_rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        config.UI_THEME["panel_bg"],
        radius=14,
        thickness=-1,
    )
    draw_rounded_rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        config.UI_THEME["accent_cyan"],
        radius=14,
        thickness=2,
    )

    # Title
    cv2.putText(
        canvas,
        "APPLICATION HELP & CONTROLS",
        (x1 + 24, y1 + 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        config.UI_THEME["text_primary"],
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (x1 + 24, y1 + 48), (x2 - 24, y1 + 48), config.UI_THEME["panel_border"], 1)

    items = [
        ("Q / ESC", "Quit and close the webcam application"),
        ("S", "Save current video frame to captures/ folder"),
        ("R", "Reset rolling smoothing history and clear output"),
        ("H", "Toggle this help screen on / off"),
        ("", ""),
        ("GUIDELINES FOR ACCURATE DETECTION:", ""),
        ("- Hold note flat inside the centered guide frame.", ""),
        ("- Ensure adequate lighting without excessive direct glare.", ""),
        ("- Avoid moving the note too fast to minimize motion blur.", ""),
        ("- The system classifies apparent denomination, not authenticity.", ""),
        ("- Non-currency objects are classified as 'Unknown / No Currency'.", ""),
    ]

    curr_y = y1 + 78
    for key, desc in items:
        if not key and not desc:
            curr_y += 10
            continue
        if key and not desc:
            cv2.putText(
                canvas,
                key,
                (x1 + 24, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                config.UI_THEME["accent_cyan"],
                1,
                cv2.LINE_AA,
            )
        elif key.startswith("-"):
            cv2.putText(
                canvas,
                key,
                (x1 + 30, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                config.UI_THEME["text_secondary"],
                1,
                cv2.LINE_AA,
            )
        else:
            # Key box
            draw_rounded_rectangle(
                canvas,
                (x1 + 24, curr_y - 14),
                (x1 + 110, curr_y + 6),
                (45, 52, 62),
                radius=4,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                key,
                (x1 + 30, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                config.UI_THEME["accent_amber"],
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                desc,
                (x1 + 125, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                config.UI_THEME["text_primary"],
                1,
                cv2.LINE_AA,
            )
        curr_y += 24

    # Footer note
    cv2.putText(
        canvas,
        "Press [H] or [SPACE] to close help",
        (x1 + 24, y2 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        config.UI_THEME["text_muted"],
        1,
        cv2.LINE_AA,
    )
