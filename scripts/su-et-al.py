import os
import argparse
from pathlib import Path

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
    "--image",
    type=str,
    required=True,
    help="raw image"
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
parser.add_argument(
    "--debug",
    type=bool,
    default=False
)
args = parser.parse_args()

print(Path.cwd())

DEBUG = args.debug
root_dir = Path(args.root_dir)
dataset_dir = root_dir / args.dataset
output_dir = Path(args.root_dir) / Path(args.output_dir)
img_path = Path(args.image)

print(f"root dir: {root_dir}")
print(f"image: {img_path}")
print(f"output dir: {output_dir}")
print(f"window size: {args.window_size}, N_min: {args.n_min}")

directions = [
    lambda x: np.roll(x, -1, axis=0),                        
    lambda x: np.roll(np.roll(x, 1, axis=1), -1, axis=0),    
    lambda x: np.roll(x, 1, axis=1),                         
    lambda x: np.roll(np.roll(x, 1, axis=1), 1, axis=0),     
]

def calculate_image_contrast(image):
    img_float = image.astype(np.float64)
    local_min = img_float.copy()
    local_max = img_float.copy()
    for roll_fn in directions:
        rolled = roll_fn(img_float)
        local_min = np.minimum(local_min, rolled)
        local_max = np.maximum(local_max, rolled)

    denominator = local_max + local_min + E
    raw_contrast = np.divide(
        (local_max - local_min),
        denominator,
        out=np.zeros_like(local_max, dtype=np.float64),
        where=denominator > E
    )
    raw_contrast = np.nan_to_num(raw_contrast, nan=0.0, posinf=0.0, neginf=0.0)
    contrast_image_normalized = cv2.normalize(raw_contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return contrast_image_normalized

# TODO(mahdi): what should be the input to the otsu?
# TODO(mahdi): ask colleague why we have a thrshould here from otsu
def calculate_threshold(image):
    _, ocimg = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ocimg

def apply_local_thresholding(image, mask, window_size, N_min):
    # TODO(mahdi): document in the report why flooring has been used here
    pad_size = window_size // 2
    I_padded = np.pad(image, pad_size, mode='reflect')
    H_padded = np.pad(mask, pad_size, mode='constant', constant_values=1)
    binarized = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi_img = I_padded[i:i+window_size, j:j+window_size]
            roi_mask = H_padded[i:i+window_size, j:j+window_size]
            # TODO(mahdi): document why 0 and not 1
            # *** this changed the results ***
            hc_vals = roi_img[roi_mask == 1]
            number_of_hc_pixels = len(hc_vals)
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
def binarize_su(image, window_size=3, N_min=9):
    contrast = calculate_image_contrast(image)
    hc_mask = calculate_threshold(contrast)
    bin = apply_local_thresholding(image, hc_mask, window_size, N_min)
    return bin, contrast, hc_mask

if __name__ == "__main__": 
    img = cv2.imread(args.image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = g.astype(np.float64) / 255.0
    if DEBUG:
        cv2.imshow("I", g)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    bin, contrast, mask = binarize_su(I, window_size=args.window_size, N_min=args.n_min)
    cv2.imwrite(os.path.join(output_dir, "0010_pr_bin_su.jpeg"), bin)
    # TODO(mahdi): masks are always black
    # cv2.imwrite(os.path.join(output_dir, "0003_hw_mask_su.jpeg"), mask)
    cv2.imwrite(os.path.join(output_dir, "0010_pr_contrast_su.jpeg"), contrast)
