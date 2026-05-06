"""
This script detects shifted images by comparing them to a reference image using phase correlation.
It processes multiple image folders in parallel and saves the results to a text file.
"""

import glob
import itertools
import os
from multiprocessing import Pool

import cv2
import numpy as np


def detect_shift(idx, total, img_dir, ref_img_path, threshold=100):
    print(f'[{idx + 1}/{total}] Working on: {img_dir}')
    # get ref image
    ref_img = cv2.imread(ref_img_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    shifted_img_list = []
    img_path_list = glob.glob(f'{img_dir}/**/*.png', recursive=True)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(gray))
        is_shifted = abs(dx) > threshold or abs(dy) > threshold
        if is_shifted:
            print(f"{img_path} may have error. dx={dx:.2f}, dy={dy:.2f}")
            shifted_img_list.append((img_path, dx, dy))
    return shifted_img_list


if __name__ == "__main__":
    target_dir = r'.\extracted_images'
    img_folders = [d for d in glob.glob(f'{target_dir}/*') if os.path.isdir(d)]

    # get ref img
    ref_img_path = glob.glob(f'{img_folders[0]}/*.png')[0]

    # apply multiprocessing
    args_list = [
        (i, len(img_folders), img_folder, ref_img_path)
        for i, img_folder in enumerate(img_folders)
    ]

    with Pool(processes=20) as pool:
        results = pool.starmap(detect_shift, args_list, chunksize=1)

    # save result
    results = list(itertools.chain(*results))
    with open("shifted_images.txt", "w") as f:
        for path, dx, dy in results:
            f.write(f"{path}, {dx:.2f}, {dy:.2f}\n")
