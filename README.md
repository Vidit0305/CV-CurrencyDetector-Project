# 💵 Indian Currency Note Classifier Using Computer Vision

A deep-learning-powered Computer Vision desktop application built with **Python**, **OpenCV**, and **TensorFlow / Keras (MobileNetV2)** that detects and classifies Indian currency banknotes in real time using your laptop webcam.

When a user presents an Indian banknote to the camera, the system recognizes its apparent denomination, applies temporal prediction smoothing to prevent jitter, performs image quality checks (blur and lighting), and displays the results on a HUD interface with live confidence and stability meters.

---

## 📌 Supported Denominations

The system classifies **six official Indian currency denominations** plus an **Unknown / Rejection** class:

| Denomination | Dominant Color Theme | Class Identifier | Official Series |
| :--- | :--- | :--- | :--- |
| **₹10** | Chocolate Brown | `10` | Mahatma Gandhi Series & New Series |
| **₹20** | Greenish Yellow | `20` | Mahatma Gandhi Series & New Series |
| **₹50** | Fluorescent Blue | `50` | Mahatma Gandhi Series & New Series |
| **₹100** | Lavender | `100` | Mahatma Gandhi Series & New Series |
| **₹200** | Bright Orange | `200` | Mahatma Gandhi New Series |
| **₹500** | Stone Grey | `500` | Mahatma Gandhi New Series |
| **Unknown** | Slate Grey | `unknown` | Hands, empty desks, cards, phones, notebooks, etc. |

> [!IMPORTANT]
> **Academic & Experimental Disclaimer**: This application is strictly an **apparent denomination classifier**. It recognizes the visual features and color palettes of banknotes. It **does NOT** verify authenticity, watermark holograms, security threads, or detect counterfeit currency.

---

## 🌟 Key Features

1. **Real-Time Webcam Classification**: Continuous frame evaluation at 30+ FPS on standard laptop CPUs without frame stutter.
2. **Transfer Learning via MobileNetV2**: High-efficiency convolutional backbone pre-trained on ImageNet with a custom classification head.
3. **Temporal Prediction Smoothing**: Rolling-window consensus algorithm (12-frame buffer) that filters out transient misclassifications, movement flicker, and hand motion.
4. **Unknown / Negative Rejection**: Dedicated 7th rejection class and confidence thresholds to avoid falsely classifying random objects, phones, books, or bare hands as banknotes.
5. **Real-Time Image Quality & Blur Checks**: Evaluates frame sharpness via Laplacian variance and checks brightness levels before inference.
6. **OpenCV HUD Interface**: Dark glassmorphic aesthetic with corner-accented guide frames, confidence meters, stability indicators, and live FPS display.
7. **Interactive Image Collection Tool**: Integrated utility (`collect_images.py`) to easily record custom samples directly from your webcam.
8. **Comprehensive Evaluation**: Standalone evaluation tool (`evaluate.py`) generating confusion matrix heatmaps, per-class metric bar charts, and sample prediction grids.

---

## 🏗️ Project Architecture & Structure

```
CV-CurrencyDetection-Project/
├── main.py                 # Real-time webcam classification application
├── train.py                # MobileNetV2 transfer learning training pipeline
├── evaluate.py             # Model evaluation, confusion matrix & metrics generator
├── collect_images.py       # Interactive webcam image collection utility
├── prepare_dataset.py      # Automated specimen downloader, synthesizer & splitter
├── config.py               # Central project configuration, colors, paths, thresholds
├── ui_helpers.py           # HUD drawing, prediction smoother & image quality checker
├── requirements.txt        # Verified Python package dependencies
├── README.md               # Complete project documentation
├── .gitignore              # Git ignore rules for virtualenvs, caches, large weights
├── dataset/                # Dataset directory
│   ├── train/              # Training split (70%)
│   ├── validation/         # Validation split (15%)
│   └── test/               # Independent test split (15%)
├── models/                 # Saved models & class mappings
│   ├── currency_mobilenet_v2.keras
│   └── class_indices.json
├── captures/               # Snapshot captures saved via [S] key
├── evaluation/             # Training logs, confusion matrix & evaluation plots
└── outputs/                # Demonstration outputs
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12**
- Working laptop webcam or external USB webcam
- Standard CPU (no dedicated GPU required)

### 2. Virtual Environment Setup

#### On Linux / macOS:
```bash
# Clone or navigate to the project directory
cd CV-CurrencyDetection-Project

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### On Windows:
```cmd
:: Navigate to project folder
cd CV-CurrencyDetection-Project

:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
.venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

You have three convenient ways to prepare or augment the dataset:

### Option A: Automated Download & Realistic Synthesis (Recommended)
Automatically download authentic specimen banknote images from Wikimedia Commons (public domain / open government specimens) and synthesize realistic lighting, angle, and background variations:
```bash
python prepare_dataset.py --download --samples-per-class 100
```

### Option B: Collect Custom Images via Webcam
Use your laptop camera to capture physical banknotes from your wallet:
```bash
python collect_images.py
```
- Press **`1` to `7`** to select the denomination (`1: ₹10`, `2: ₹20`, `3: ₹50`, `4: ₹100`, `5: ₹200`, `6: ₹500`, `7: unknown`).
- Press **`SPACE`** or **`C`** to capture single images.
- Press **`B`** to toggle continuous burst mode.
- Press **`T`** to cycle destination split (`AUTO [70/15/15]`, `train`, `validation`, `test`).
- Press **`Q`** or **`ESC`** to quit.

### Option C: Import an Existing Image Folder
If you already have a folder of currency images organized by denomination:
```bash
python prepare_dataset.py --split /path/to/my_folder
```

To verify the integrity and counts of the current dataset:
```bash
python prepare_dataset.py --verify
```

---

## 🧠 Model Training

Train the **MobileNetV2** classifier using transfer learning:
```bash
python train.py
```

### Training Highlights:
- **Phase 1 (Head Training)**: Trains the custom classification head with frozen MobileNetV2 base weights.
- **Phase 2 (Fine-Tuning)**: Unfreezes the top 30 layers with a lower learning rate (`1e-5`) for fine-grained feature adaptation.
- **Data Augmentation**: Random rotations, zooming, translations, and flips applied on the fly to prevent overfitting.
- **Automated Checkpoints**: Saves the best model weights based on `val_accuracy` to `models/currency_mobilenet_v2.keras`.
- **Training Curves**: Automatically plots and saves training vs. validation accuracy and loss to `evaluation/training_history.png`.

Optional training flags:
```bash
# Custom epochs or batch size
python train.py --epochs 10 --fine-tune-epochs 2 --batch-size 16 --force
```

---

## 🎯 Model Evaluation

Evaluate the saved model on the independent `dataset/test/` set:
```bash
python evaluate.py
```

This generates:
- **`evaluation/confusion_matrix.png`**: High-resolution confusion matrix heatmap.
- **`evaluation/per_class_metrics.png`**: Precision, Recall, and F1-score bar chart per denomination.
- **`evaluation/sample_predictions.png`**: Visual grid of test predictions with confidence and correctness color coding.
- **`evaluation/classification_report.txt`**: Precision, recall, and support metrics for each class.

---

## 📷 Real-Time Webcam Detection

To launch the live webcam classification interface:
```bash
python main.py
```

### 🎮 Keyboard Controls

| Key | Action | Details |
| :---: | :--- | :--- |
| **`Q`** or **`ESC`** | **Quit** | Closes camera stream, destroys OpenCV windows, and exits safely. |
| **`S`** | **Save Snapshot** | Saves the current camera frame to `captures/` with a timestamp and displays a toast banner. |
| **`R`** | **Reset Smoothing** | Clears the rolling prediction buffer and resets the displayed state. |
| **`H`** | **Toggle Help** | Displays or hides the on-screen help and instructions modal. |

### 💡 Tips for Best Results:
1. **Positioning**: Hold the banknote flat inside the centered corner-bracket guide frame.
2. **Lighting**: Ensure even ambient lighting; avoid heavy direct glare or reflections.
3. **Motion**: Hold the note steady for a fraction of a second to allow temporal smoothing to confirm the denomination.
4. **Distance**: Hold the note so it occupies 60%–85% of the guide box.

---

## 🛠️ Configuration & Customization

All thresholds and parameters can be adjusted in [`config.py`](config.py):
- `CONFIDENCE_THRESHOLD = 0.70`: Minimum model confidence required to display a positive identification.
- `SMOOTHING_WINDOW_SIZE = 12`: Number of recent frames tracked in the rolling consensus buffer.
- `STABILITY_MIN_COUNT = 7`: Minimum agreeing votes required in the buffer.
- `INFERENCE_INTERVAL = 2`: Runs model inference every N frames to ensure high-FPS camera rendering.
- `BLUR_THRESHOLD = 50.0`: Laplacian variance threshold below which "Blur Detected" warning triggers.

---

## 📄 License & Credits
- **Model Architecture**: MobileNetV2 (Sandler et al., Google Research).
- **Specimen Images**: Public domain and open government specimen releases hosted on Wikimedia Commons.
- **Project Type**: Educational / Academic Computer Vision Project.
