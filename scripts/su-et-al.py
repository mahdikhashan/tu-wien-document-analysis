import os
import argparse
from glob import glob
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.metrics import peak_signal_noise_ratio as psnr

E = 1e-7

parser = argparse.ArgumentParser(description="binarize image using su-et-l method")
parser.add_argument("--root-dir", type=str, default="", help="root like ../")
parser.add_argument(
    "--dataset",
    type=str,
    default="data",
    help="specific dataset path"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="data/processed",
    help="dir of saved processed image"
)
parser.add_argument(
    "--result-dir",
    type=str,
    default="results",
    help="dir to save evaluation results"
)
parser.add_argument(
    "--models-dir",
    type=str,
    default="models",
    help="dir of models"
)
parser.add_argument(
    "--hw-folder",
    type=str,
    help="handwritten images for test ex. DIBC02009_Test_images-handwritten"
)
parser.add_argument(
    "--pr-folder",
    type=str,
    help="printed images for test ex. DIBCO2009_Test_images-printed"
)
parser.add_argument(
    "--window-size",
    type=int,
    default=20,
    help="window size"
)
parser.add_argument(
    "--n-min",
    type=float,
    default=1,
    help="n-min parameter"
)
args = parser.parse_args()

print(Path.cwd())

root_dir = Path(args.root_dir)
dataset_dir = root_dir / args.dataset
image_dir_hw = dataset_dir / args.hw_folder
image_dir_pr = dataset_dir / args.pr_folder
output_dir = Path(args.root_dir) / Path(args.output_dir)
results_dir = Path(args.root_dir) / Path(args.result_dir)
model_dir = Path(args.root_dir) / Path(args.models_dir)

print(f"root dir: {root_dir}")
print(f"handwritten images: {image_dir_hw}")
print(f"printed images: {image_dir_pr}")
print(f"output dir: {output_dir}")
print(f"results dir: {results_dir}")
print(f"models dir: {model_dir}")
print(f"window size: {args.window_size}, N_min: {args.n_min}")

# TODO(mahdi): use a dataset class when want to have evaluation with more algorithms
all_hw = sorted(glob(os.path.join(image_dir_hw, "*.tif")))
all_pr = sorted(glob(os.path.join(image_dir_pr, "*.tif")))

image_paths_hw = [p for p in all_hw if "_gt" not in p]
gt_paths_hw    = [p for p in all_hw if "_gt" in p]

image_paths_pr = [p for p in all_pr if "_gt" not in p]
gt_paths_pr    = [p for p in all_pr if "_gt" in p]

os.makedirs(results_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def calculate_image_contrast(image, win_size=3):
    from scipy.ndimage import maximum_filter, minimum_filter
    pad_size = win_size // 2
    padded_image = np.pad(image, pad_size, mode="reflect")
    f_min = minimum_filter(padded_image, size=win_size)[pad_size:-pad_size, pad_size:-pad_size]
    f_max = maximum_filter(padded_image, size=win_size)[pad_size:-pad_size, pad_size:-pad_size]
    contrast_image = (f_max - f_min) / (f_max + f_min + E)
    # TODO(mahdi): add to the report why this is needed
    contrast_image_normalized = cv2.normalize(contrast_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return contrast_image_normalized

# TODO(mahdi): what should be the input to the otsu?
# TODO(mahdi): ask colleague why we have a thrshould here from otsu
def calculate_threshold(image):
    from skimage.filters import threshold_otsu
    otsu_thresh_value = threshold_otsu(image)
    # print(otsu_thresh_value)
    return np.where(image > otsu_thresh_value, 0, 1).astype(np.uint8)

def apply_local_thresholding(image, mask, window_size, N_min):
    # TODO(mahdi): document in the report why flooring has been used here
    pad_size = window_size // 2
    I_padded = np.pad(image, pad_size, mode='reflect')  # pad image
    H_padded = np.pad(mask, pad_size, mode='constant', constant_values=1)  # pad high contrast mask
    binarized = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi_img = I_padded[i:i+window_size, j:j+window_size]
            roi_mask = H_padded[i:i+window_size, j:j+window_size]
            # TODO(mahdi): document why 0 and not 1
            hc_vals = roi_img[roi_mask == 0]
            number_of_hc_pixels = len(hc_vals)
            # print("hc_vals", number_of_hc_pixels)
            # TODO(mahdi): handle forground as well
            if number_of_hc_pixels >= N_min:
                mean = hc_vals.mean()
                std = hc_vals.std()
                local_threshold = mean + std / 2
                if image[i, j] <= local_threshold:
                    binarized[i, j] = 255
                else:
                    binarized[i, j] = 0
            else:
                binarized[i, j] = 0

    return binarized

# TODO(mahdi): add an invert arg if required, communicate it
def binarize_su(image, window_size=25, N_min=6):
    contrast = calculate_image_contrast(image)
    hc_mask = calculate_threshold(contrast)
    bin = apply_local_thresholding(image, hc_mask, window_size, N_min)
    return bin, contrast, hc_mask

if __name__ == "__main__": 
    results_hw = {}
    images_hw = {
        os.path.basename(img_path)[:-4]: (
            cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
            cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        )
        for img_path, gt_path in zip(image_paths_hw, gt_paths_hw)
    }
    images_pr = {
        os.path.basename(img_path)[:-4]: (
            cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
            cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        )
        for img_path, gt_path in zip(image_paths_pr, gt_paths_pr)
    }
    for _, (name, (img, img_gt)) in enumerate(tqdm((images_pr | images_hw).items())):
        bin_img, contrast, mask = binarize_su(img, window_size=args.window_size, N_min=args.n_min)
        cv2.imwrite(os.path.join(output_dir, name + "_hw_bin_su.jpeg"), bin_img)
        cv2.imwrite(os.path.join(output_dir, name + "_hw_contrast_su.jpeg"), contrast)
        cv2.imwrite(os.path.join(output_dir, name + "_hw_mask_su.jpeg"), mask)

        gt_bin = (img_gt < 128).astype(np.uint8)
        pred_bin = (bin_img > 128).astype(np.uint8)

        results_hw[name] = {
            "F1": f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0),
            "Precision": precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0),
            "Recall": recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0),
            "PSNR": psnr(img_gt, bin_img, data_range=255)
        }
