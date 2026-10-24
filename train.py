"""
Training Pipeline for Indian Currency Note Classifier.

Uses MobileNetV2 transfer learning with two-phase training:
- Phase 1: Train custom classification head with frozen feature extractor.
- Phase 2: Fine-tune top layers with lower learning rate for peak accuracy.

Evaluates performance on held-out test set, saves evaluation plots,
confusion matrix, and classification report.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config


def verify_dataset_structure(force: bool = False) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """Check dataset folders and count images per class."""
    print("\n" + "=" * 65)
    print(" [1/5] VERIFYING DATASET STRUCTURE & CLASS INTEGRITY")
    print("=" * 65)

    if not config.DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory {config.DATASET_DIR} does not exist.")
        print("   Please run: python prepare_dataset.py --download")
        return False, {}

    counts: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
    has_insufficient_data = False
    total_samples = 0

    header = f"{'Class':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Status'}"
    print(header)
    print("-" * len(header))

    for cls in config.CLASS_NAMES:
        c_tr = len(list((config.TRAIN_DIR / cls).glob("*.[jJpP][pPnN][gG]"))) if (config.TRAIN_DIR / cls).exists() else 0
        c_va = len(list((config.VAL_DIR / cls).glob("*.[jJpP][pPnN][gG]"))) if (config.VAL_DIR / cls).exists() else 0
        c_te = len(list((config.TEST_DIR / cls).glob("*.[jJpP][pPnN][gG]"))) if (config.TEST_DIR / cls).exists() else 0

        counts["train"][cls] = c_tr
        counts["validation"][cls] = c_va
        counts["test"][cls] = c_te
        total_samples += (c_tr + c_va + c_te)

        if c_tr == 0 or c_va == 0 or c_te == 0:
            status = "MISSING ❌"
            has_insufficient_data = True
        elif c_tr < 10:
            status = "LOW DATA ⚠️"
            has_insufficient_data = True
        else:
            status = "READY ✅"

        disp = f"₹{cls}" if cls != "unknown" else "unknown"
        print(f"{disp:<12} | {c_tr:<8} | {c_va:<8} | {c_te:<8} | {status}")

    print("-" * len(header))
    print(f"Total Dataset Images: {total_samples}")

    if has_insufficient_data and not force:
        print("\n⚠️ WARNING: Some classes have missing or very low data (< 10 images in train).")
        print("   For best results, run: python prepare_dataset.py --download")
        print("   Or collect images with: python collect_images.py")
        proceed = input("Do you still wish to proceed with training? [y/N]: ").strip().lower()
        if proceed != "y":
            return False, counts

    return True, counts


def load_datasets(batch_size: int = config.BATCH_SIZE):
    """Load train, validation, and test datasets with tf.data pipelines."""
    print("\n" + "=" * 65)
    print(" [2/5] LOADING & PREPROCESSING DATASETS")
    print("=" * 65)

    # Consistent class ordering matching config.CLASS_NAMES
    classes = config.CLASS_NAMES

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        class_names=classes,
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        class_names=classes,
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        class_names=classes,
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    # Save class indices mapping
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_mapping = {
        "classes": classes,
        "class_to_idx": {cls: i for i, cls in enumerate(classes)},
        "idx_to_class": {i: cls for i, cls in enumerate(classes)},
        "display_names": {i: config.CLASS_METADATA[cls]["display_name"] for i, cls in enumerate(classes)},
    }
    with open(config.CLASS_INDICES_PATH, "w") as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Saved class mapping to: {config.CLASS_INDICES_PATH}")

    # Optimize pipeline throughput
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, classes


def build_model(num_classes: int = len(config.CLASS_NAMES)) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Construct MobileNetV2 transfer learning model with integrated preprocessing
    and data augmentation.
    """
    print("\n" + "=" * 65)
    print(" [3/5] BUILDING MOBILENETV2 TRANSFER LEARNING ARCHITECTURE")
    print("=" * 65)

    # Base pretrained model (ImageNet weights)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # Data augmentation layers (active only during training)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="image_input")
    x = data_augmentation(inputs)
    # Built-in MobileNetV2 preprocessing maps [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="IndianCurrencyClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Base model: MobileNetV2 (Frozen, {len(base_model.layers)} layers)")
    print(f"Classification head: GlobalAvgPool -> BatchNorm -> Dense(256) -> Dropout({config.DROPOUT_RATE}) -> Dense({num_classes}, softmax)")
    print(f"Total params: {model.count_params():,}")

    return model, base_model


def train_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int = config.INITIAL_EPOCHS,
    fine_tune_epochs: int = config.FINE_TUNE_EPOCHS,
):
    """Execute two-phase training: Head training + Top-layer fine tuning."""
    print("\n" + "=" * 65)
    print(f" [4/5] TRAINING MODEL (Phase 1: Head Training - {epochs} Epochs)")
    print("=" * 65)

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_SAVE_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # Phase 2: Fine-Tuning Top Layers of MobileNetV2
    if fine_tune_epochs > 0:
        print("\n" + "-" * 60)
        print(f" Phase 2: Fine-Tuning Top Layers ({fine_tune_epochs} Epochs)")
        print("-" * 60)

        base_model.trainable = True
        # Freeze all layers except the top 30 layers
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(f" Unfrozen {len(base_model.layers) - fine_tune_at} layers for fine-tuning.")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_phase2 = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.MODEL_SAVE_PATH),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        total_epochs = len(history1.history["accuracy"]) + fine_tune_epochs
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=len(history1.history["accuracy"]),
            epochs=total_epochs,
            callbacks=callbacks_phase2,
            verbose=1,
        )

        # Merge histories
        merged_history = {
            "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
            "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
            "loss": history1.history["loss"] + history2.history["loss"],
            "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
        }
    else:
        merged_history = history1.history

    return model, merged_history


def plot_and_save_history(history: dict):
    """Plot and save training/validation accuracy and loss curves."""
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    history_json_path = config.EVALUATION_DIR / "training_history.json"
    history_plot_path = config.EVALUATION_DIR / "training_history.png"

    # Save JSON log
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_json_path, "w") as f:
        json.dump(serializable_history, f, indent=4)

    # Plot Accuracy & Loss
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, history["accuracy"], "o-", color="#1f77b4", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], "s--", color="#ff7f0e", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, history["loss"], "o-", color="#d62728", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "s--", color="#2ca02c", label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss Over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    print(f"Saved training history curves to: {history_plot_path}")


def evaluate_final_model(model: tf.keras.Model, test_ds, classes: list):
    """Evaluate model on independent test dataset, plot confusion matrix and save report."""
    print("\n" + "=" * 65)
    print(" [5/5] INDEPENDENT TEST SET EVALUATION")
    print("=" * 65)

    # Collect ground-truth and predictions
    y_true_indices = []
    y_pred_indices = []

    print("Running batch inference on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_indices.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_indices.extend(np.argmax(preds, axis=1))

    y_true_indices = np.array(y_true_indices)
    y_pred_indices = np.array(y_pred_indices)

    # Compute test accuracy
    test_accuracy = float(np.mean(y_true_indices == y_pred_indices))
    print(f"\n🎯 FINAL TEST SET ACCURACY: {test_accuracy * 100:.2f}%\n")

    # Classification Report
    display_names = [config.CLASS_METADATA[cls]["display_name"] for cls in classes]
    report_str = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=display_names,
        digits=4,
        zero_division=0,
    )
    print("Detailed Classification Report:\n")
    print(report_str)

    report_path = config.EVALUATION_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("INDIAN CURRENCY NOTE CLASSIFIER - INDEPENDENT TEST SET EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
        f.write(report_str)
    print(f"Saved classification report to: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    cm_path = config.EVALUATION_DIR / "confusion_matrix.png"

    plt.figure(figsize=(9, 7.5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Dataset", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    short_labels = [f"₹{cls}" if cls != "unknown" else "Unknown" for cls in classes]
    plt.xticks(tick_marks, short_labels, rotation=45, fontsize=10)
    plt.yticks(tick_marks, short_labels, fontsize=10)

    # Annotate numbers inside matrix cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:d}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save model metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "MobileNetV2",
        "input_shape": config.INPUT_SHAPE,
        "classes": classes,
        "test_accuracy": test_accuracy,
        "model_file": str(config.MODEL_SAVE_PATH.name),
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {config.MODEL_METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Indian Currency Note Classifier")
    parser.add_argument("--epochs", type=int, default=config.INITIAL_EPOCHS, help=f"Head training epochs (default: {config.INITIAL_EPOCHS})")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help=f"Fine-tuning epochs (default: {config.FINE_TUNE_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--force", action="store_true", help="Force training without interactive confirmation prompt")
    args = parser.parse_args()

    start_time = time.time()
    ready, counts = verify_dataset_structure(force=args.force)
    if not ready:
        sys.exit(1)

    train_ds, val_ds, test_ds, classes = load_datasets(batch_size=args.batch_size)
    model, base_model = build_model(num_classes=len(classes))
    model, history = train_model(
        model,
        base_model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )

    plot_and_save_history(history)

    # Reload best saved model weights for final evaluation
    if config.MODEL_SAVE_PATH.exists():
        print(f"\nReloading best saved model from {config.MODEL_SAVE_PATH}...")
        best_model = tf.keras.models.load_model(str(config.MODEL_SAVE_PATH))
    else:
        best_model = model

    evaluate_final_model(best_model, test_ds, classes)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY in {elapsed/60:.1f} minutes!")
    print(f"   Model saved at: {config.MODEL_SAVE_PATH}")
    print(f"   Launch real-time webcam detector with: python main.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
