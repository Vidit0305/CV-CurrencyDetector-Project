"""
Main Real-Time Webcam Application for Indian Currency Note Classifier.

Features:
- Live laptop webcam video stream with high-performance inference throttling.
- MobileNetV2 transfer learning model classification across 6 Indian denominations + Unknown class.
- Rolling window temporal prediction smoothing and consensus stabilization.
- Real-time image quality evaluation (blur and exposure detection).
- High-contrast, dark-mode HUD with corner guide reticles, status badges, and animated confidence meters.
- Keyboard controls:
    [Q] / [ESC] : Quit application safely.
    [S]         : Save current frame snapshot to captures/ folder.
    [R]         : Reset prediction smoothing and displayed state.
    [H]         : Toggle help instructions overlay.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Suppress TensorFlow C++ engine debug logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import config
import ui_helpers


class CurrencyClassifierApp:
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.classes = config.CLASS_NAMES
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.smoother = ui_helpers.PredictionSmoother()

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = config.DEFAULT_CAMERA_INDEX

        # Inference throttling state
        self.frame_counter = 0
        self.last_predicted_class: Optional[str] = None
        self.last_raw_confidence: float = 0.0

        # UI state
        self.toast_message: Optional[str] = None
        self.toast_expiry: float = 0.0
        self.show_help = False

        # FPS measurement
        self.fps_timer = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

    def check_and_load_model(self) -> bool:
        """Verify model file exists and load it into memory."""
        print("\n" + "=" * 65)
        print(" INITIALIZING INDIAN CURRENCY NOTE CLASSIFIER")
        print("=" * 65)

        if not config.MODEL_SAVE_PATH.exists():
            print(f"\n❌ ERROR: Trained model file not found at:")
            print(f"   {config.MODEL_SAVE_PATH.resolve()}")
            print("\nTo train the model before running the webcam detector:")
            print("   1. Generate or prepare dataset:")
            print("      python prepare_dataset.py --download")
            print("   2. Train the MobileNetV2 model:")
            print("      python train.py")
            print("   3. Once training finishes, launch:")
            print("      python main.py\n")
            return False

        print(f"Loading trained neural network from:\n  {config.MODEL_SAVE_PATH} ...", end=" ", flush=True)
        try:
            self.model = tf.keras.models.load_model(str(config.MODEL_SAVE_PATH))
            print("Done ✅")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            return False

        # Load class mapping if available
        if config.CLASS_INDICES_PATH.exists():
            try:
                with open(config.CLASS_INDICES_PATH, "r") as f:
                    mapping = json.load(f)
                self.classes = mapping.get("classes", config.CLASS_NAMES)
                raw_idx_to_class = mapping.get("idx_to_class", {})
                self.idx_to_class = {int(k): v for k, v in raw_idx_to_class.items()}
                print(f"Loaded class mapping: {list(self.idx_to_class.values())}")
            except Exception as e:
                print(f"Warning: Could not parse class indices file ({e}), using default classes.")

        # Warm up model with a dummy inference to avoid lag on first camera frame
        dummy_input = np.zeros((1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("Model inference engine warmed up and ready.")
        return True

    def open_camera(self) -> bool:
        """Find and open accessible camera feed with fallback indices."""
        indices = [self.camera_index] + [i for i in config.CAMERA_FALLBACK_INDICES if i != self.camera_index]

        for idx in indices:
            print(f"Testing camera index {idx}...", end=" ", flush=True)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DESIRED_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DESIRED_FRAME_HEIGHT)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.camera_index = idx
                    h, w = frame.shape[:2]
                    print(f"Connected ✅ ({w}x{h})")
                    return True
                cap.release()
            print("Unavailable")

        print("\n❌ Error: No accessible webcam could be initialized.")
        print("   - Ensure your webcam is connected.")
        print("   - Check that no other application (Zoom, browser, etc.) is locking the camera.")
        return False

    def predict_roi(self, roi_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Preprocess ROI and generate prediction using MobileNetV2 model.
        Returns:
            (predicted_class_name, confidence)
        """
        # Convert BGR to RGB
        rgb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_roi, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Batch dimension: shape (1, 224, 224, 3)
        input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)

        # Generate prediction
        probabilities = self.model.predict(input_tensor, verbose=0)[0]
        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])
        predicted_class = self.idx_to_class.get(top_idx, "unknown")

        return predicted_class, confidence

    def save_snapshot(self, frame: np.ndarray) -> str:
        """Save high-resolution snapshot of current video frame to captures folder."""
        config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        save_path = config.CAPTURES_DIR / filename

        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        exact_path = str(save_path.resolve())
        print(f"\n📸 Snapshot saved to: {exact_path}")

        self.toast_message = f"Snapshot saved: {filename}"
        self.toast_expiry = time.time() + 2.2
        return exact_path

    def update_fps(self):
        """Calculate running frames per second."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 0.5:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_timer = time.time()

    def run(self):
        """Main application loop."""
        if not self.check_and_load_model():
            return

        if not self.open_camera():
            return

        window_name = "Indian Currency Note Classifier - Live Webcam"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print(" WEBCAM CLASSIFIER RUNNING")
        print(" Controls:")
        print("   [Q] / [ESC] : Quit application")
        print("   [S]         : Save current frame to captures/")
        print("   [R]         : Reset prediction smoothing")
        print("   [H]         : Toggle help instructions")
        print("=" * 65 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: Camera frame could not be read.")
                    time.sleep(0.04)
                    continue

                # Mirror frame horizontally for natural intuitive webcam view
                frame = cv2.flip(frame, 1)
                self.update_fps()
                self.frame_counter += 1

                h, w = frame.shape[:2]
                roi_coords = ui_helpers.get_roi_coordinates(w, h)
                x1, y1, x2, y2 = roi_coords
                roi = frame[y1:y2, x1:x2].copy()

                # Evaluate image quality (blur and brightness checks)
                quality_info = ui_helpers.check_image_quality(roi)

                # Periodic model inference for smooth frame rate
                if self.frame_counter % config.INFERENCE_INTERVAL == 0:
                    if quality_info["is_valid"] and not quality_info["is_blurry"]:
                        raw_class, raw_conf = self.predict_roi(roi)
                        self.last_predicted_class = raw_class
                        self.last_raw_confidence = raw_conf
                        # Feed prediction into rolling smoother
                        self.smoother.update(raw_class, raw_conf)
                    else:
                        # Image is too blurry or dark, treat conservatively
                        self.smoother.update("unknown", 0.40)

                # Get stabilized consensus results
                stable_class = self.smoother.current_stable_class
                confidence = self.smoother.current_confidence
                stability_ratio = self.smoother.current_stability_ratio
                status_state = self.smoother.status_state

                # Check active toast message
                active_toast = self.toast_message if time.time() < self.toast_expiry else None

                # Render guide reticle over ROI
                ui_helpers.draw_roi_guide(frame, roi_coords)

                # Render complete HUD interface
                display_frame = ui_helpers.draw_hud(
                    frame=frame,
                    stable_class=stable_class,
                    confidence=confidence,
                    stability_ratio=stability_ratio,
                    status_state=status_state,
                    fps=self.current_fps,
                    quality_info=quality_info,
                    toast_message=active_toast,
                    show_help=self.show_help,
                )

                cv2.imshow(window_name, display_frame)

                # Handle keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("Exiting Indian Currency Note Classifier...")
                    break
                elif key in (ord("s"), ord("S")):    # Save snapshot
                    self.save_snapshot(frame)
                elif key in (ord("r"), ord("R")):    # Reset smoothing
                    self.smoother.reset()
                    print("Prediction smoothing buffer reset.")
                elif key in (ord("h"), ord("H")):    # Toggle help
                    self.show_help = not self.show_help

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Webcam released. All OpenCV windows closed.")


def main():
    app = CurrencyClassifierApp()
    app.run()


if __name__ == "__main__":
    main()
