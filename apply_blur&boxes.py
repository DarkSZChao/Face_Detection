import glob
import os
from multiprocessing import Pool

import cv2


def process_subfolder(idx, total, input_sub_dir, output_dir):
    print(f"[{idx + 1}/{total}] Working on: [{input_sub_dir}] to [{output_dir}]")

    for img_path in glob.glob(f"{input_sub_dir}/**/*.png", recursive=True):
        label_path = f'{os.path.dirname(img_path)}/{os.path.basename(img_path).split(".")[0]}.txt'

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Ensure the coordinates are within image bounds
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                # apply gaussian blur to the image
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # apply a green box to the face area (for debugging)
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (99, 99), 30)  # apply blur to the face area

        output_path = os.path.join(output_dir, os.path.relpath(img_path, os.path.dirname(input_sub_dir)))  # keep the input folder structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    input_dir = r'.\marked_results'
    output_dir = r'F:\blurred_images'

    input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/*') if os.path.isdir(d)]

    args_list = [
        (i, len(input_sub_dir_list), input_sub_dir, output_dir)
        for i, input_sub_dir in enumerate(input_sub_dir_list)
    ]

    with Pool(processes=20) as pool:
        pool.starmap(process_subfolder, args_list, chunksize=1)
