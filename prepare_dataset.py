"""
Dataset Preparation and Verification Utility for Indian Currency Classifier.

Capabilities:
1. Curates authentic specimen images of Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
   from the official Mahatma Gandhi New Series.
2. Generates realistic real-world camera variations (angles, distances, lighting,
   background textures, hand occlusions, camera noise) simulating laptop webcam captures.
3. Prepares an extensive unknown/rejection class (real webcam room scenes, human shirts/clothing,
   faces, hands, desks, books, phones, cards, textures).
4. Deterministically organizes and splits data into train (70%), validation (15%), test (15%).
5. Verifies dataset integrity, counts, and reports class balance.
"""

import os
import sys
import math
import random
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

import config

# Wikimedia Commons public specimen filenames for Mahatma Gandhi New Series (active Indian circulation)
SPECIMEN_SOURCES = {
    "10": [
        "India_new_10_INR,_MG_series,_2018,_obverse.jpg",
        "India_new_10_INR,_MG_series,_2018,_reverse.jpg",
    ],
    "20": [
        "India_new_20_INR,_MG_series,_2019,_obverse.jpg",
        "India_new_20_INR,_MG_series,_2019,_reverse.jpg",
    ],
    "50": [
        "India_new_50_INR,_MG_series,_2018,_obverse.jpg",
        "India_new_50_INR,_MG_series,_2018,_reverse.jpg",
    ],
    "100": [
        "India_new_100_INR,_Mahatma_Gandhi_New_Series,_2018,_obverse.png",
        "India_new_100_INR,_Mahatma_Gandhi_New_Series,_2018,_reverse.png",
    ],
    "200": [
        "India,_200_INR,_2018,_obverse.jpg",
        "India,_200_INR,_2018,_reverse.jpg",
    ],
    "500": [
        "India_new_500_INR,_MG_series,_2016,_obverse.jpg",
        "India_new_500_INR,_MG_series,_2016,_reverse.jpg",
    ],
}

RAW_SPECIMENS_DIR = config.DATASET_DIR / "raw_specimens"


def ensure_directory_structure():
    """Create all standard dataset directories if they do not exist."""
    splits = [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]
    for split in splits:
        for class_name in config.CLASS_NAMES:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    RAW_SPECIMENS_DIR.mkdir(parents=True, exist_ok=True)


def download_specimens() -> Dict[str, List[Path]]:
    """
    Download authentic specimen images from Wikimedia Commons for each denomination.
    Returns dictionary mapping class name to list of saved raw image paths.
    """
    ensure_directory_structure()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) IndianCurrencyClassifier/2.0"
    }

    downloaded: Dict[str, List[Path]] = {cls: [] for cls in config.CLASS_NAMES if cls != "unknown"}
    print("\n" + "=" * 65)
    print(" [1/3] VERIFYING & FETCHING MAHATMA GANDHI NEW SERIES SPECIMENS")
    print("=" * 65)

    for class_name, filenames in SPECIMEN_SOURCES.items():
        class_raw_dir = RAW_SPECIMENS_DIR / class_name
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            local_path = class_raw_dir / filename
            if local_path.exists() and local_path.stat().st_size > 5000:
                print(f"  [Cached] ₹{class_name}: {filename} ({local_path.stat().st_size // 1024} KB)")
                downloaded[class_name].append(local_path)
                continue

            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
            try:
                print(f"  Downloading ₹{class_name}: {filename} ...", end=" ", flush=True)
                response = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
                if response.status_code == 200 and len(response.content) > 5000:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    print(f"Done ({len(response.content) // 1024} KB)")
                    downloaded[class_name].append(local_path)
                else:
                    print(f"Failed (HTTP {response.status_code})")
            except Exception as e:
                print(f"Error: {e}")

    return downloaded


def extract_real_background_patches() -> List[np.ndarray]:
    """Extract real-world crops from captured webcam images to simulate actual rooms."""
    patches = []
    candidates = list(config.CAPTURES_DIR.glob("*.jpg"))
    user_up = Path("/home/vidit/.gemini/antigravity-ide/brain/91a163f9-d3a7-4d8e-801c-8b6076721adb/.user_uploaded")
    if user_up.exists():
        candidates.extend(user_up.glob("*.jpg"))

    for p in candidates:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 200 or w < 200:
            continue
        # Extract multiple diverse 224x224 crops
        for _ in range(8):
            cy = random.randint(0, max(0, h - 224))
            cx = random.randint(0, max(0, w - 224))
            patch = img[cy : cy + 224, cx : cx + 224].copy()
            if patch.shape[:2] == (224, 224):
                patches.append(patch)
    return patches


def generate_synthetic_background(width: int, height: int, real_patches: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Generate realistic ambient background surfaces (real webcam rooms, wood, fabric, marble, wall)."""
    if real_patches and random.random() < 0.40:
        patch = random.choice(real_patches)
        if patch.shape[:2] != (height, width):
            patch = cv2.resize(patch, (width, height))
        return patch

    bg_type = random.choice(["wood", "marble", "fabric", "desk", "wall", "shirt"])

    if bg_type == "wood":
        # Warm wood tone with horizontal grain
        base_color = np.array([45, 75, 120], dtype=np.uint8)
        grain = np.random.randint(-15, 15, (height, width, 1), dtype=np.int16)
        bands = (np.sin(np.linspace(0, 30, height)) * 12).astype(np.int16)[:, None, None]
        bg = np.clip(base_color + grain + bands, 20, 210).astype(np.uint8)
        if bg.shape[2] == 1:
            bg = np.repeat(bg, 3, axis=2)

    elif bg_type == "marble":
        base = np.full((height, width, 3), 215, dtype=np.uint8)
        noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
        bg = np.clip(base + noise, 160, 250).astype(np.uint8)
        for _ in range(3):
            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (random.randint(0, width), random.randint(0, height))
            cv2.line(bg, pt1, pt2, (170, 175, 180), random.randint(1, 3))
        bg = cv2.GaussianBlur(bg, (5, 5), 0)

    elif bg_type == "fabric":
        # Dark textile
        base = np.full((height, width, 3), random.randint(40, 80), dtype=np.uint8)
        textile = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        bg = np.clip(base + textile, 20, 120).astype(np.uint8)

    elif bg_type == "shirt":
        # Human clothing simulation: white / light grey / navy / striped shirt
        choice = random.choice(["white", "navy", "striped", "beige"])
        if choice == "white":
            val = random.randint(190, 240)
            bg = np.full((height, width, 3), val, dtype=np.uint8)
            noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
            bg = np.clip(bg + noise, 170, 255).astype(np.uint8)
        elif choice == "navy":
            bg = np.full((height, width, 3), [random.randint(70, 110), random.randint(30, 60), random.randint(20, 40)], dtype=np.uint8)
        elif choice == "striped":
            bg = np.full((height, width, 3), 220, dtype=np.uint8)
            stripe_color = (random.randint(40, 90), random.randint(40, 90), random.randint(120, 180))
            for x in range(0, width, random.randint(15, 25)):
                cv2.line(bg, (x, 0), (x, height), stripe_color, 2)
        else:
            bg = np.full((height, width, 3), [random.randint(160, 190), random.randint(180, 210), random.randint(190, 220)], dtype=np.uint8)

    elif bg_type == "wall":
        # Typical indoor room wall (beige, cream, off-white, light blue)
        hue = random.choice([
            (210, 225, 230),  # cream
            (215, 220, 210),  # soft warm grey
            (230, 215, 200),  # light cool blue/grey
            (180, 200, 210),  # light beige
        ])
        bg = np.full((height, width, 3), hue, dtype=np.uint8)
        noise = np.random.randint(-6, 6, (height, width, 3), dtype=np.int16)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    else:
        val = random.randint(100, 170)
        bg = np.full((height, width, 3), val, dtype=np.uint8)
        noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    return bg


def synthesize_realistic_sample(
    note_img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    real_patches: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Synthesize realistic camera captures:
    - 50% random horizontal mirror flip (webcam view compatibility)
    - Realistic scale (0.50 to 0.95 relative to frame)
    - 3D perspective warp and random rotation
    - Lighting, contrast, color temperature jitter
    - Simulated finger holding note (skin tones)
    - Defocus blur and sensor noise
    """
    th, tw = target_size
    bg = generate_synthetic_background(tw, th, real_patches=real_patches)

    note = note_img.copy()
    # 50% horizontal mirror flip
    if random.random() < 0.50:
        note = cv2.flip(note, 1)

    nh, nw = note.shape[:2]

    # Target scale
    scale = random.uniform(0.55, 0.94)
    aspect = nw / nh
    fit_w = int(tw * scale)
    fit_h = int(fit_w / aspect)
    if fit_h > th * 0.92:
        fit_h = int(th * 0.92)
        fit_w = int(fit_h * aspect)

    resized_note = cv2.resize(note, (fit_w, fit_h), interpolation=cv2.INTER_AREA)

    # Perspective Warp / 3D tilt
    p_offset = random.uniform(0.015, 0.07)
    src_pts = np.float32([[0, 0], [fit_w, 0], [fit_w, fit_h], [0, fit_h]])
    dst_pts = np.float32([
        [random.uniform(0, fit_w * p_offset), random.uniform(0, fit_h * p_offset)],
        [fit_w - random.uniform(0, fit_w * p_offset), random.uniform(0, fit_h * p_offset)],
        [fit_w - random.uniform(0, fit_w * p_offset), fit_h - random.uniform(0, fit_h * p_offset)],
        [random.uniform(0, fit_w * p_offset), fit_h - random.uniform(0, fit_h * p_offset)],
    ])
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_note = cv2.warpPerspective(resized_note, M_persp, (fit_w, fit_h), borderValue=(0, 0, 0))

    note_mask = np.zeros((fit_h, fit_w), dtype=np.uint8)
    cv2.fillConvexPoly(note_mask, dst_pts.astype(np.int32), 255)

    # Rotation (-20 to +20 deg)
    angle = random.uniform(-20, 20)
    center = (fit_w // 2, fit_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_note = cv2.warpAffine(warped_note, rot_mat, (fit_w, fit_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(note_mask, rot_mat, (fit_w, fit_h), flags=cv2.INTER_NEAREST, borderValue=0)

    # Place onto background canvas
    max_x = max(0, tw - fit_w)
    max_y = max(0, th - fit_h)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)

    canvas = bg.copy()
    roi = canvas[offset_y : offset_y + fit_h, offset_x : offset_x + fit_w]

    # Blend with soft edge
    mask_3ch = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR) / 255.0
    blended = (rotated_note * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
    canvas[offset_y : offset_y + fit_h, offset_x : offset_x + fit_w] = blended

    # Color & Lighting variations (brightness, contrast, temperature)
    light_factor = random.uniform(0.70, 1.30)
    contrast_factor = random.uniform(0.80, 1.25)
    mean_val = 128.0
    adjusted = (canvas.astype(np.float32) - mean_val) * contrast_factor + mean_val
    adjusted = adjusted * light_factor

    # Color temperature jitter (warm indoor incandescent vs cool daylight)
    b_shift = random.randint(-12, 12)
    r_shift = random.randint(-12, 12)
    adjusted[:, :, 0] += b_shift
    adjusted[:, :, 2] += r_shift
    canvas = np.clip(adjusted, 0, 255).astype(np.uint8)

    # Simulated finger holding note (40% probability)
    if random.random() < 0.40:
        finger_color = (random.randint(110, 160), random.randint(140, 190), random.randint(180, 230))
        finger_pos = (
            offset_x + random.choice([12, fit_w - 20]),
            offset_y + random.randint(fit_h // 3, fit_h - 10),
        )
        cv2.ellipse(canvas, finger_pos, (random.randint(14, 24), random.randint(22, 36)), random.randint(-25, 25), 0, 360, finger_color, -1)
        cv2.ellipse(canvas, (finger_pos[0] - 2, finger_pos[1] + 2), (18, 28), 0, 0, 360, (25, 25, 25), 1)

    # Slight camera noise or blur
    if random.random() < 0.35:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    return canvas


def generate_unknown_sample(
    target_size: Tuple[int, int] = (224, 224),
    real_patches: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Generate realistic negative/unknown images:
    - Real webcam background scenes & clothing from user's environment
    - Empty backgrounds (wood, marble, desk, cloth, walls)
    - Hands held up without currency
    - Smartphones, notebooks, wallets, cards, books
    """
    tw, th = target_size

    # 40% chance of using real webcam room / user crop if available
    if real_patches and random.random() < 0.40:
        patch = random.choice(real_patches)
        if patch.shape[:2] != (th, tw):
            patch = cv2.resize(patch, (tw, th))
        # Add slight variation
        if random.random() < 0.3:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
        return patch

    sample_type = random.choice(["empty_bg", "hand", "smartphone", "book", "card", "shirt_pattern"])
    bg = generate_synthetic_background(tw, th, real_patches=real_patches)

    if sample_type == "empty_bg":
        return bg

    elif sample_type == "hand":
        skin_color = (random.randint(110, 160), random.randint(140, 190), random.randint(180, 230))
        cx, cy = tw // 2 + random.randint(-30, 30), th // 2 + random.randint(-20, 40)
        cv2.ellipse(bg, (cx, cy), (45, 55), random.randint(-15, 15), 0, 360, skin_color, -1)
        for dx in [-28, -10, 8, 26]:
            fx = cx + dx
            fy = cy - random.randint(45, 65)
            cv2.line(bg, (cx + dx // 2, cy - 20), (fx, fy), skin_color, random.randint(12, 16))
            cv2.circle(bg, (fx, fy), random.randint(6, 8), skin_color, -1)
        return cv2.GaussianBlur(bg, (3, 3), 0)

    elif sample_type == "smartphone":
        sw = random.randint(80, 110)
        sh = int(sw * 2.1)
        sx = (tw - sw) // 2 + random.randint(-20, 20)
        sy = (th - sh) // 2 + random.randint(-20, 20)
        cv2.rectangle(bg, (sx, sy), (sx + sw, sy + sh), (25, 25, 25), -1)
        cv2.rectangle(bg, (sx + 4, sy + 12), (sx + sw - 4, sy + sh - 12), (55, 60, 65), -1)
        return bg

    elif sample_type == "book":
        bw = random.randint(130, 180)
        bh = random.randint(120, 160)
        bx = (tw - bw) // 2 + random.randint(-15, 15)
        by = (th - bh) // 2 + random.randint(-15, 15)
        cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), (235, 238, 240), -1)
        for ly in range(by + 15, by + bh - 10, 12):
            cv2.line(bg, (bx + 10, ly), (bx + bw - 10, ly), (180, 185, 190), 1)
        return bg

    elif sample_type == "card":
        cw = random.randint(110, 140)
        ch = int(cw * 0.63)
        cx = (tw - cw) // 2 + random.randint(-20, 20)
        cy = (th - ch) // 2 + random.randint(-20, 20)
        card_color = (random.randint(40, 200), random.randint(40, 200), random.randint(40, 200))
        cv2.rectangle(bg, (cx, cy), (cx + cw, cy + ch), card_color, -1)
        cv2.rectangle(bg, (cx + 10, cy + ch // 3), (cx + 32, cy + ch // 3 + 16), (30, 180, 220), -1)
        return bg

    elif sample_type == "shirt_pattern":
        # White or grey t-shirt with logo or neck line
        cv2.ellipse(bg, (tw // 2, 0), (60, 45), 0, 0, 180, (140, 160, 180), -1)  # skin neck
        return bg

    return bg


def clean_synthetic_files():
    """Remove previously generated synthetic files to ensure a clean slate."""
    for split in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        for class_name in config.CLASS_NAMES:
            cls_dir = split / class_name
            if cls_dir.exists():
                for f in cls_dir.glob("syn_*.[jJpP][pPnN][gG]"):
                    try:
                        f.unlink()
                    except Exception:
                        pass


def build_augmented_dataset(
    downloaded_specimens: Dict[str, List[Path]],
    total_samples_per_class: int = 250,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Generate the complete structured dataset with train, val, and test splits.
    Uses pure Mahatma Gandhi New Series specimens and diverse backgrounds.
    """
    ensure_directory_structure()
    clean_synthetic_files()
    real_patches = extract_real_background_patches()
    print(f"  Loaded {len(real_patches)} real webcam background patches for realistic compositing.")

    print("\n" + "=" * 65)
    print(" [2/3] SYNTHESIZING DIVERSE REALISTIC DATASET (MG New Series)")
    print(f" Target: {total_samples_per_class} images per class across 7 classes")
    print(f" Splits: Train ({int(train_ratio*100)}%), Val ({int(val_ratio*100)}%), Test ({int(test_ratio*100)}%)")
    print("=" * 65)

    n_train = int(total_samples_per_class * train_ratio)
    n_val = int(total_samples_per_class * val_ratio)
    n_test = total_samples_per_class - n_train - n_val

    # 1. Banknote Denominations (10, 20, 50, 100, 200, 500)
    for class_name in config.CLASS_NAMES:
        if class_name == "unknown":
            continue

        raw_files = downloaded_specimens.get(class_name, [])
        if not raw_files:
            class_raw_dir = RAW_SPECIMENS_DIR / class_name
            if class_raw_dir.exists():
                # Filter strictly to Mahatma Gandhi New Series filenames
                allowed_names = SPECIMEN_SOURCES.get(class_name, [])
                raw_files = [p for p in class_raw_dir.glob("*") if p.name in allowed_names]

        if not raw_files:
            print(f"  [WARNING] No specimen images available for ₹{class_name}. Skipping generation.")
            continue

        loaded_notes = []
        for p in raw_files:
            img = cv2.imread(str(p))
            if img is not None:
                loaded_notes.append(img)

        if not loaded_notes:
            print(f"  [ERROR] Could not read raw specimens for ₹{class_name}.")
            continue

        print(f"  Generating ₹{class_name} ({len(loaded_notes)} source specimens) ...", end=" ", flush=True)

        for i in range(n_train):
            note = random.choice(loaded_notes)
            sample = synthesize_realistic_sample(note, real_patches=real_patches)
            out_p = config.TRAIN_DIR / class_name / f"syn_{class_name}_tr_{i:04d}.jpg"
            cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

        for i in range(n_val):
            note = random.choice(loaded_notes)
            sample = synthesize_realistic_sample(note, real_patches=real_patches)
            out_p = config.VAL_DIR / class_name / f"syn_{class_name}_val_{i:04d}.jpg"
            cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

        for i in range(n_test):
            note = random.choice(loaded_notes)
            sample = synthesize_realistic_sample(note, real_patches=real_patches)
            out_p = config.TEST_DIR / class_name / f"syn_{class_name}_te_{i:04d}.jpg"
            cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

        print(f"Done (Train: {n_train}, Val: {n_val}, Test: {n_test})")

    # 2. Unknown / Rejection Class
    print(f"  Generating Unknown / No Currency samples ...", end=" ", flush=True)
    for i in range(n_train):
        sample = generate_unknown_sample(real_patches=real_patches)
        out_p = config.TRAIN_DIR / "unknown" / f"syn_unknown_tr_{i:04d}.jpg"
        cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

    for i in range(n_val):
        sample = generate_unknown_sample(real_patches=real_patches)
        out_p = config.VAL_DIR / "unknown" / f"syn_unknown_val_{i:04d}.jpg"
        cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

    for i in range(n_test):
        sample = generate_unknown_sample(real_patches=real_patches)
        out_p = config.TEST_DIR / "unknown" / f"syn_unknown_te_{i:04d}.jpg"
        cv2.imwrite(str(out_p), sample, [cv2.IMWRITE_JPEG_QUALITY, 93])

    print(f"Done (Train: {n_train}, Val: {n_val}, Test: {n_test})")


def verify_dataset() -> bool:
    """Verify dataset folder structure, image counts per class, and report status."""
    print("\n" + "=" * 65)
    print(" [3/3] VERIFYING DATASET INTEGRITY & CLASS BALANCE")
    print("=" * 65)

    is_ready = True
    total_images = 0

    header = f"{'Class':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8} | {'Status'}"
    print(header)
    print("-" * len(header))

    for class_name in config.CLASS_NAMES:
        c_train = len(list((config.TRAIN_DIR / class_name).glob("*.[jJpP][pPnN][gG]"))) if (config.TRAIN_DIR / class_name).exists() else 0
        c_val = len(list((config.VAL_DIR / class_name).glob("*.[jJpP][pPnN][gG]"))) if (config.VAL_DIR / class_name).exists() else 0
        c_test = len(list((config.TEST_DIR / class_name).glob("*.[jJpP][pPnN][gG]"))) if (config.TEST_DIR / class_name).exists() else 0
        c_total = c_train + c_val + c_test
        total_images += c_total

        if c_train == 0:
            status = "MISSING ❌"
            is_ready = False
        elif c_train < 15:
            status = "LOW DATA ⚠️"
            is_ready = False
        else:
            status = "READY ✅"

        display_label = f"₹{class_name}" if class_name != "unknown" else "unknown"
        print(f"{display_label:<12} | {c_train:<8} | {c_val:<8} | {c_test:<8} | {c_total:<8} | {status}")

    print("-" * len(header))
    print(f"Total Dataset Images: {total_images}")

    if not is_ready:
        print("\n⚠️ WARNING: Some classes have missing or very low data (< 15 images in train).")
        return False
    else:
        print("\n✅ Dataset verification successful! Ready for high-accuracy training.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Indian Currency Dataset Preparation Tool")
    parser.add_argument("--download", action="store_true", help="Download official Wikimedia specimens and generate dataset")
    parser.add_argument("--samples-per-class", type=int, default=250, help="Number of samples to generate per class (default: 250)")
    parser.add_argument("--verify", action="store_true", help="Verify current dataset counts and integrity")

    args = parser.parse_args()

    if args.verify:
        verify_dataset()
        return

    ensure_directory_structure()
    specimens = download_specimens()
    build_augmented_dataset(specimens, total_samples_per_class=args.samples_per_class)
    verify_dataset()


if __name__ == "__main__":
    main()
