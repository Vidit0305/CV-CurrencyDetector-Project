"""
Webcam Image Collection Utility for Indian Currency Classifier.

Allows users to collect custom training, validation, and test images
using their laptop webcam for all six Indian currency denominations
and the Unknown / No Currency rejection class.

Controls:
    1-7       : Select class (1: 10, 2: 20, 3: 50, 4: 100, 5: 200, 6: 500, 7: unknown)
    TAB       : Cycle to next class
    SPACE / C : Capture single frame inside guide ROI
    B         : Toggle Burst Capture Mode (continuous capture every 500ms)
    T         : Toggle target split (AUTO [70% train / 15% val / 15% test], or manual TRAIN / VAL / TEST)
    Q / ESC   : Quit and return to terminal
"""

import sys
import time
import random
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

import config
import ui_helpers

# Ensure directory structure exists
for split in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
    for cls in config.CLASS_NAMES:
        (split / cls).mkdir(parents=True, exist_ok=True)


class ImageCollectorApp:
    def __init__(self, camera_index: int = config.DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.classes = config.CLASS_NAMES
        self.current_class_idx = 0  # Starts at '10'

        # Target split: 'AUTO', 'train', 'validation', 'test'
        self.split_modes = ["AUTO", "train", "validation", "test"]
        self.current_split_idx = 0

        # Burst capture mode
        self.burst_active = False
        self.last_burst_time = 0.0
        self.burst_interval = 0.45  # seconds

        # Session counters
        self.session_counts = {cls: 0 for cls in self.classes}
        self.toast_msg = None
        self.toast_expiry = 0.0

    @property
    def current_class(self) -> str:
        return self.classes[self.current_class_idx]

    @property
    def current_split_mode(self) -> str:
        return self.split_modes[self.current_split_idx]

    def get_disk_count(self, class_name: str) -> int:
        """Count total images saved on disk for this class across all splits."""
        total = 0
        for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
            p = split_dir / class_name
            if p.exists():
                total += len(list(p.glob("*.[jJpP][pPnN][gG]")))
        return total

    def open_camera(self) -> bool:
        """Initialize webcam with fallback indices."""
        for idx in [self.camera_index] + [i for i in config.CAMERA_FALLBACK_INDICES if i != self.camera_index]:
            print(f"Attempting to open camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Configure resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DESIRED_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DESIRED_FRAME_HEIGHT)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    self.camera_index = idx
                    print(f"✅ Successfully opened camera index {idx} ({test_frame.shape[1]}x{test_frame.shape[0]})")
                    return True
                cap.release()

        print("❌ Error: No accessible webcam found. Ensure webcam is connected and not used by another app.")
        return False

    def save_capture(self, frame: np.ndarray, roi_coords: tuple) -> str:
        """Crop and save image to selected class directory."""
        x1, y1, x2, y2 = roi_coords
        roi_img = frame[y1:y2, x1:x2].copy()

        # Decide split folder
        if self.current_split_mode == "AUTO":
            r = random.random()
            if r < 0.70:
                target_split = config.TRAIN_DIR
                split_name = "train"
            elif r < 0.85:
                target_split = config.VAL_DIR
                split_name = "validation"
            else:
                target_split = config.TEST_DIR
                split_name = "test"
        else:
            split_name = self.current_split_mode
            target_split = config.DATASET_DIR / split_name

        class_dir = target_split / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"capture_{self.current_class}_{split_name}_{timestamp}.jpg"
        save_path = class_dir / filename

        # Embed with genuine banknote aspect ratio preserved
        rh, rw = roi_img.shape[:2]
        tw = int(224 * 0.75)
        th = max(1, int(tw * (rh / rw)))
        if th > 220:
            th = 220
            tw = int(th * (rw / rh))
        resized = cv2.resize(roi_img, (tw, th), interpolation=cv2.INTER_AREA)
        processed = np.full((224, 224, 3), 128, dtype=np.uint8)
        py = (224 - th) // 2
        px = (224 - tw) // 2
        processed[py : py + th, px : px + tw] = resized
        cv2.imwrite(str(save_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.session_counts[self.current_class] += 1
        print(f"📸 Saved [{self.current_class.upper()}] ({split_name}): {save_path}")

        self.toast_msg = f"Captured ₹{self.current_class} -> {split_name}/ ({self.session_counts[self.current_class]} this session)"
        self.toast_expiry = time.time() + 1.8
        return str(save_path)

    def draw_collector_hud(self, frame: np.ndarray, roi_coords: tuple) -> np.ndarray:
        """Render HUD specific to dataset collection."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Header Bar
        header_h = 56
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), config.UI_THEME["bg_dark"], -1)
        cv2.line(overlay, (0, header_h), (w, header_h), config.UI_THEME["panel_border"], 1)
        cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)

        cv2.circle(canvas, (24, 28), 6, config.UI_THEME["accent_green"], -1)
        cv2.putText(
            canvas,
            "DATASET IMAGE COLLECTOR - WEBCAM UTILITY",
            (40, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            config.UI_THEME["text_primary"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Capture custom banknote images with automated train / val / test splitting",
            (42, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            config.UI_THEME["text_secondary"],
            1,
            cv2.LINE_AA,
        )

        # 2. Guide ROI
        ui_helpers.draw_roi_guide(canvas, roi_coords, accent_color=config.UI_THEME["accent_green"])

        # 3. Bottom Control Card
        card_h = 130
        y1 = h - card_h - 18
        y2 = h - 18
        x1 = 20
        x2 = w - 20

        card_overlay = canvas.copy()
        ui_helpers.draw_rounded_rectangle(card_overlay, (x1, y1), (x2, y2), config.UI_THEME["panel_bg"], radius=12)
        cv2.addWeighted(card_overlay, 0.90, canvas, 0.10, 0, canvas)
        ui_helpers.draw_rounded_rectangle(canvas, (x1, y1), (x2, y2), config.UI_THEME["panel_border"], radius=12, thickness=1)

        # Selected Class Indicator
        cls_key = self.current_class
        meta = config.CLASS_METADATA.get(cls_key, config.CLASS_METADATA["unknown"])
        class_label = f"CURRENT CLASS: {meta['display_name']}"
        cv2.putText(
            canvas,
            class_label,
            (x1 + 20, y1 + 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            meta["color_bgr"],
            2,
            cv2.LINE_AA,
        )

        # Stats
        session_cnt = self.session_counts[cls_key]
        total_disk = self.get_disk_count(cls_key)
        stats_str = f"Captured this session: {session_cnt}  |  Total on disk: {total_disk} images"
        cv2.putText(
            canvas,
            stats_str,
            (x1 + 20, y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            config.UI_THEME["text_primary"],
            1,
            cv2.LINE_AA,
        )

        # Split & Burst Badges on Right
        badge_x = x2 - 340
        split_badge_text = f"SPLIT: {self.current_split_mode.upper()} [T]"
        ui_helpers.draw_rounded_rectangle(canvas, (badge_x, y1 + 18), (badge_x + 150, y1 + 44), (50, 60, 75), radius=6)
        cv2.putText(canvas, split_badge_text, (badge_x + 12, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

        burst_bg = config.UI_THEME["accent_red"] if self.burst_active else (50, 60, 75)
        burst_text = f"BURST: {'ON' if self.burst_active else 'OFF'} [B]"
        ui_helpers.draw_rounded_rectangle(canvas, (badge_x + 160, y1 + 18), (badge_x + 310, y1 + 44), burst_bg, radius=6)
        cv2.putText(canvas, burst_text, (badge_x + 172, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Controls Footer
        controls_str = "[1-7] Select Class  |  [TAB] Next  |  [SPACE/C] Capture  |  [B] Burst  |  [T] Split  |  [Q] Exit"
        cv2.putText(canvas, controls_str, (x1 + 20, y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.UI_THEME["text_muted"], 1, cv2.LINE_AA)

        # Toast notification
        if self.toast_msg and time.time() < self.toast_expiry:
            tw, _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
            tx = (w - tw) // 2
            ui_helpers.draw_rounded_rectangle(canvas, (tx - 16, header_h + 12), (tx + tw + 16, header_h + 46), config.UI_THEME["accent_green"], radius=8)
            cv2.putText(canvas, self.toast_msg, (tx, header_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1, cv2.LINE_AA)

        return canvas

    def run(self):
        """Main execution loop for image collector."""
        if not self.open_camera():
            return

        window_name = "Indian Currency Classifier - Image Collection Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 60)
        print(" IMAGE COLLECTOR READY")
        print(" Align banknote inside guide box and press [SPACE] to capture.")
        print(" Press [1-7] to select denomination (1: 10, 2: 20, 3: 50, 4: 100, 5: 200, 6: 500, 7: unknown)")
        print("=" * 60)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: Camera frame not received.")
                    time.sleep(0.05)
                    continue

                # Mirror frame horizontally for intuitive interaction
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                roi_coords = ui_helpers.get_roi_coordinates(w, h)

                # Burst capture handling
                now = time.time()
                if self.burst_active and (now - self.last_burst_time >= self.burst_interval):
                    self.save_capture(frame, roi_coords)
                    self.last_burst_time = now

                # Render HUD
                display = self.draw_collector_hud(frame, roi_coords)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("\nExiting Image Collector...")
                    break
                elif key in (ord(" "), ord("c"), ord("C")):  # Capture
                    self.save_capture(frame, roi_coords)
                elif key in (ord("b"), ord("B")):  # Toggle burst
                    self.burst_active = not self.burst_active
                    print(f"Burst Mode: {'ACTIVE' if self.burst_active else 'OFF'}")
                elif key in (ord("t"), ord("T")):  # Toggle split
                    self.current_split_idx = (self.current_split_idx + 1) % len(self.split_modes)
                    print(f"Target Split: {self.current_split_mode}")
                elif key == 9:  # TAB key
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
                    print(f"Selected Class: ₹{self.current_class}")
                elif ord("1") <= key <= ord("7"):
                    self.current_class_idx = key - ord("1")
                    print(f"Selected Class: ₹{self.current_class}")

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Webcam released. Windows closed successfully.")


if __name__ == "__main__":
    collector = ImageCollectorApp()
    collector.run()
