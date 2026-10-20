"""
Evaluation Utility for Indian Currency Note Classifier.

Evaluates a trained MobileNetV2 model against the test dataset:
- Computes overall accuracy, precision, recall, F1-score.
- Generates high-resolution confusion matrix heatmap.
- Generates per-class performance bar chart.
- Generates a visual sample prediction grid showing predictions with confidence.
- Saves detailed results in the evaluation/ folder.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config


def load_model_and_classes():
    """Load trained Keras model and class mapping indices."""
    if not config.MODEL_SAVE_PATH.exists():
        print(f"\n❌ Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("   Please train the model first by running:")
        print("   python train.py\n")
        sys.exit(1)

    print(f"Loading trained model from {config.MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(str(config.MODEL_SAVE_PATH))

    if config.CLASS_INDICES_PATH.exists():
        with open(config.CLASS_INDICES_PATH, "r") as f:
            mapping = json.load(f)
        classes = mapping.get("classes", config.CLASS_NAMES)
    else:
        classes = config.CLASS_NAMES

    return model, classes


def evaluate():
    model, classes = load_model_and_classes()
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    if not config.TEST_DIR.exists():
        print(f"❌ Error: Test dataset directory {config.TEST_DIR} does not exist.")
        sys.exit(1)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        class_names=classes,
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    y_true_indices = []
    y_pred_indices = []
    y_pred_probs = []
    sample_images = []
    sample_true = []
    sample_pred = []
    sample_conf = []

    print("Running evaluation on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        batch_true = np.argmax(labels.numpy(), axis=1)
        batch_pred = np.argmax(preds, axis=1)
        batch_conf = np.max(preds, axis=1)

        y_true_indices.extend(batch_true)
        y_pred_indices.extend(batch_pred)
        y_pred_probs.extend(preds)

        # Collect first 12 images for sample visual grid
        if len(sample_images) < 12:
            for i in range(len(batch_true)):
                if len(sample_images) < 12:
                    sample_images.append(images[i].numpy().astype(np.uint8))
                    sample_true.append(batch_true[i])
                    sample_pred.append(batch_pred[i])
                    sample_conf.append(batch_conf[i])

    y_true_indices = np.array(y_true_indices)
    y_pred_indices = np.array(y_pred_indices)

    # 1. Overall Accuracy
    accuracy = float(np.mean(y_true_indices == y_pred_indices))
    print("\n" + "=" * 65)
    print(f"🎯 TEST SET ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 65)

    display_names = [config.CLASS_METADATA[c]["display_name"] for c in classes]

    # 2. Classification Report
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=display_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=display_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:\n")
    print(report_text)

    with open(config.EVALUATION_DIR / "classification_report.txt", "w") as f:
        f.write("INDIAN CURRENCY NOTE CLASSIFIER - TEST SET EVALUATION\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report_text)

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(9, 7.5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.1f}%)", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    short_labels = [f"₹{cls}" if cls != "unknown" else "Unknown" for cls in classes]
    plt.xticks(tick_marks, short_labels, rotation=45, fontsize=10)
    plt.yticks(tick_marks, short_labels, fontsize=10)

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
    cm_path = config.EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix to: {cm_path}")

    # 4. Per-Class Metrics Bar Chart
    precisions = [report[d]["precision"] * 100 for d in display_names]
    recalls = [report[d]["recall"] * 100 for d in display_names]
    f1_scores = [report[d]["f1-score"] * 100 for d in display_names]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width, precisions, width, label="Precision", color="#3b82f6")
    plt.bar(x, recalls, width, label="Recall", color="#10b981")
    plt.bar(x + width, f1_scores, width, label="F1-Score", color="#f59e0b")

    plt.xlabel("Currency Class", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Per-Class Classification Metrics (Precision / Recall / F1)", fontsize=13, fontweight="bold")
    plt.xticks(x, short_labels, fontsize=10)
    plt.ylim(0, 110)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    metrics_path = config.EVALUATION_DIR / "per_class_metrics.png"
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"✅ Saved per-class metrics bar chart to: {metrics_path}")

    # 5. Visual Sample Predictions Grid
    if sample_images:
        fig, axes = plt.subplots(3, 4, figsize=(13, 10))
        fig.suptitle(f"Sample Test Predictions (Model Accuracy: {accuracy*100:.1f}%)", fontsize=14, fontweight="bold")

        for idx, ax in enumerate(axes.flat):
            if idx < len(sample_images):
                img = sample_images[idx]
                t_label = short_labels[sample_true[idx]]
                p_label = short_labels[sample_pred[idx]]
                conf = sample_conf[idx] * 100
                is_correct = sample_true[idx] == sample_pred[idx]

                ax.imshow(img)
                title_color = "#16a34a" if is_correct else "#dc2626"
                ax.set_title(
                    f"True: {t_label} | Pred: {p_label}\nConf: {conf:.1f}%",
                    fontsize=9.5,
                    fontweight="bold",
                    color=title_color,
                )
            ax.axis("off")

        plt.tight_layout()
        samples_path = config.EVALUATION_DIR / "sample_predictions.png"
        plt.savefig(samples_path, dpi=300)
        plt.close()
        print(f"✅ Saved sample predictions grid to: {samples_path}")

    print("\nEvaluation completed successfully.\n")


if __name__ == "__main__":
    evaluate()
