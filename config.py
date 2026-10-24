"""
Configuration settings for the Indian Currency Note Classifier.

Contains class labels, model architecture parameters, paths,
and real-time webcam inference parameters.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
CAPTURES_DIR = BASE_DIR / "captures"
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Model File Paths
MODEL_SAVE_PATH = MODELS_DIR / "currency_mobilenet_v2.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Supported Currency Denominations + Unknown rejection class
# Ordered logically
CLASS_NAMES = ["10", "20", "50", "100", "200", "500", "unknown"]

# Clean Class Metadata for UI and Reporting (Strict ASCII for reliable OpenCV font rendering)
CLASS_METADATA = {
    "10": {
        "display_name": "Rs. 10 NOTE",
        "description": "Ten Rupee Banknote (Chocolate Brown)",
        "color_bgr": (42, 82, 135),    # Chocolate brown
        "color_rgb": (135, 82, 42),
        "accent_hex": "#87522A",
    },
    "20": {
        "display_name": "Rs. 20 NOTE",
        "description": "Twenty Rupee Banknote (Greenish Yellow)",
        "color_bgr": (34, 180, 180),   # Greenish yellow
        "color_rgb": (180, 180, 34),
        "accent_hex": "#B4B422",
    },
    "50": {
        "display_name": "Rs. 50 NOTE",
        "description": "Fifty Rupee Banknote (Fluorescent Blue)",
        "color_bgr": (220, 150, 0),    # Fluorescent blue
        "color_rgb": (0, 150, 220),
        "accent_hex": "#0096DC",
    },
    "100": {
        "display_name": "Rs. 100 NOTE",
        "description": "One Hundred Rupee Banknote (Lavender)",
        "color_bgr": (195, 120, 150),  # Lavender
        "color_rgb": (150, 120, 195),
        "accent_hex": "#9678C3",
    },
    "200": {
        "display_name": "Rs. 200 NOTE",
        "description": "Two Hundred Rupee Banknote (Bright Orange)",
        "color_bgr": (20, 135, 245),   # Bright orange
        "color_rgb": (245, 135, 20),
        "accent_hex": "#F58714",
    },
    "500": {
        "display_name": "Rs. 500 NOTE",
        "description": "Five Hundred Rupee Banknote (Stone Grey)",
        "color_bgr": (135, 140, 135),  # Stone grey
        "color_rgb": (135, 140, 135),
        "accent_hex": "#878C87",
    },
    "unknown": {
        "display_name": "NO CURRENCY DETECTED",
        "description": "Non-currency object or empty background",
        "color_bgr": (100, 105, 110),  # Neutral slate
        "color_rgb": (110, 105, 100),
        "accent_hex": "#6E6964",
    },
}

# Image & Model Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Training Hyperparameters
BATCH_SIZE = 16
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
DROPOUT_RATE = 0.30

# Screen & Display Settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FULLSCREEN_DEFAULT = True

# Real-Time Webcam Parameters
DEFAULT_CAMERA_INDEX = 0
CAMERA_FALLBACK_INDICES = [0, 1, 2]
DESIRED_FRAME_WIDTH = 1280
DESIRED_FRAME_HEIGHT = 720

# Region of Interest (ROI) Guide Box (centered 2.1:1 banknote aspect ratio)
ROI_WIDTH = 560
ROI_HEIGHT = 270

# Banknote Texture & Presence Thresholds
MIN_NOTE_EDGE_DENSITY = 0.0035    # Calibrated for indoor webcams (plain walls are ~0.000, notes are 0.008-0.020)
CONFIDENCE_THRESHOLD = 0.40       # Robust positive classification threshold for 7-class softmax
UNKNOWN_CONFIDENCE_THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.08           # Minimum probability margin between top and runner-up class

# Image Quality Check Thresholds
BLUR_THRESHOLD = 30.0            # Laplacian variance threshold
MIN_BRIGHTNESS = 25.0            # Permissive to indoor ambient lighting
MAX_BRIGHTNESS = 245.0           # Mean pixel value above which scene is overexposed

# UI Visual Palette (BGR for OpenCV)
UI_THEME = {
    "bg_dark": (18, 20, 24),
    "panel_bg": (28, 32, 38),
    "panel_border": (60, 68, 78),
    "text_primary": (245, 245, 248),
    "text_secondary": (165, 175, 185),
    "text_muted": (110, 118, 128),
    "accent_cyan": (230, 175, 45),
    "accent_green": (85, 205, 100),
    "accent_amber": (50, 180, 245),
    "accent_red": (75, 80, 235),
    "roi_guide": (240, 200, 70),
}
