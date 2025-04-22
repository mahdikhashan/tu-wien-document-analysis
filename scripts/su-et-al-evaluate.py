import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import precision_score, recall_score, f1_score


def load_and_prepare_image(image_path: Path):
    if not image_path.is_file():
        print(f"Error: Image file not found at {image_path}")
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return None

    unique_values = np.unique(img)
    is_binary_0_255 = np.all(np.isin(unique_values, [0, 255]))

    if not is_binary_0_255:
        print(
            f"Warning: Image {image_path.name} is not strictly binary (0/255). Values: {unique_values}. Applying threshold (>127)."
        )
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return img.astype(np.uint8)


def calculate_metrics(pred_bin: np.ndarray | None, gt_bin: np.ndarray | None):
    if pred_bin is None or gt_bin is None:
        return

    # if pred_bin.shape != gt_bin.shape:
    #     print(
    #         f"Warning: Resizing GT shape {gt_bin.shape} to match Pred shape {pred_bin.shape}"
    #     )
    #     try:
    #         gt_bin = cv2.resize(
    #             gt_bin,
    #             (pred_bin.shape[1], pred_bin.shape[0]),
    #             interpolation=cv2.INTER_NEAREST,
    #         )
    #         # Ensure resize keeps it binary
    #         gt_bin = np.where(gt_bin > 127, 255, 0).astype(np.uint8)
    #     except Exception as e:
    #         print(f"Error during resizing: {e}")
    #         return None, None, None, None

    try:
        pred_flat_01 = (pred_bin.flatten() / 255).astype(np.uint8)
        gt_flat_01 = (gt_bin.flatten() / 255).astype(np.uint8)

        # P, R, F1 (assuming 1 is foreground)
        precision = precision_score(gt_flat_01, pred_flat_01, zero_division=0)
        recall = recall_score(gt_flat_01, pred_flat_01, zero_division=0)
        f1 = f1_score(gt_flat_01, pred_flat_01, zero_division=0)

        # Calculate PSNR on 0-255 images
        psnr = peak_signal_noise_ratio(gt_bin, pred_bin, data_range=255)

        return precision, recall, f1, psnr

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a prediction image against a ground truth image."
    )
    parser.add_argument(
        "--prediction",
        "-p",
        type=str,
        required=True,
        help="Path to the predicted binary image (PNG/BMP etc.).",
    )
    parser.add_argument(
        "--groundtruth",
        "-g",
        type=str,
        required=True,
        help="Path to the ground truth binary image.",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the results in a CSV file.",
    )

    args = parser.parse_args()

    pred_path = Path(args.prediction)
    gt_path = Path(args.groundtruth)
    output_csv_path = Path(args.output_csv) if args.output_csv else None

    print(f"Evaluating Prediction: {pred_path.name}")
    print(f"Against Ground Truth: {gt_path.name}")

    pred_image = load_and_prepare_image(pred_path)
    pred_image = 255 - pred_image
    gt_image = load_and_prepare_image(gt_path)

    precision, recall, f1, psnr = calculate_metrics(pred_image, gt_image)

    print("-" * 30)
    if all(metric is not None for metric in [precision, recall, f1, psnr]):
        print("Evaluation Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  PSNR:      {psnr:.4f}")
    else:
        print("Metric calculation failed.")
    print("-" * 30)

    if output_csv_path and all(
        metric is not None for metric in [precision, recall, f1, psnr]
    ):
        results = {
            "prediction_image": [pred_path.name],
            "groundtruth_image": [gt_path.name],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
            "psnr": [psnr],
        }
        results_df = pd.DataFrame(results)
        try:
            # Ensure output directory exists if path includes directories
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_csv_path, index=False, mode="w", header=True)
            print(f"Results saved to: {output_csv_path}")
        except Exception as e:
            print(f"Error saving results to CSV {output_csv_path}: {e}")
    elif output_csv_path:
        print(f"Skipping CSV saving due to metric calculation errors.")

    print("Evaluation script finished.")
