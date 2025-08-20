import glob
import itertools
from multiprocessing import Pool

import cv2
import numpy as np


def detect_shift(img_dir, ref_img_path, threshold=100):
    print(f'Working on: {img_dir}')
    # get ref image
    ref_img = cv2.imread(ref_img_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    shifted_img_list = []
    img_path_list = glob.glob(f'{img_dir}/**/*.png')
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
    img_folders = glob.glob('./dataset_pending')
    img_folders = [
        folder for folder in img_folders
        if 0 <= int(folder.split('\\')[-1]) <= 900
    ]
    img_folders = sorted(img_folders, key=lambda x: int(x.split('\\')[-1]))  # sort the folder

    # get ref img
    ref_img_path = glob.glob(f'./dataset_pending/123/**/123_0_116.png')[0]

    # apply multiprocessing
    args_list = [(img_folder, ref_img_path) for img_folder in img_folders]

    with Pool(processes=16) as pool:
        results = pool.starmap(detect_shift, args_list)

    # save result
    results = list(itertools.chain(*results))
    with open("shifted_images.txt", "w") as f:
        for path, dx, dy in results:
            f.write(f"{path}, {dx:.2f}, {dy:.2f}\n")
