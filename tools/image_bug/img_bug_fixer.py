"""
This script identifies images listed in a "shifted_images.txt" file and replaces them with their nearest non-shifted neighbors (previous or next image in the same folder).
If both neighbors are also shifted, it logs a warning.
"""

import os
import glob
import shutil

def load_shifted_list(txt_path):
    shifted = set()
    with open(txt_path, "r") as f:
        for line in f:
            path = line.strip().split(",")[0]
            shifted.add(os.path.normpath(path))
    return shifted


def replace_from_txt(shifted_txt):
    shifted_set = load_shifted_list(shifted_txt)

    for img_path in shifted_set:
        folder = os.path.dirname(img_path)

        if not os.path.exists(img_path):
            print(f"⚠️ File not found: {img_path}")
            continue

        # 获取当前文件夹所有图片并排序
        imgs = sorted(glob.glob(os.path.join(folder, "*.png")))

        try:
            idx = imgs.index(img_path)
        except ValueError:
            print(f"⚠️ Not in list: {img_path}")
            continue

        replacement = None

        # ✅ 优先前一帧
        if idx > 0:
            prev_img = os.path.normpath(imgs[idx - 1])
            if prev_img not in shifted_set:
                replacement = imgs[idx - 1]

        # ✅ 否则用后一帧
        if replacement is None and idx < len(imgs) - 1:
            next_img = os.path.normpath(imgs[idx + 1])
            if next_img not in shifted_set:
                replacement = imgs[idx + 1]

        if replacement:
            print(f"Replacing:\n  {img_path}\n→ {replacement}")
            shutil.copy2(replacement, img_path)
        else:
            print(f"⚠️ WARNING: Cannot replace {img_path} (neighbors also bad)")


if __name__ == "__main__":
    shifted_txt = "shifted_images.txt"
    replace_from_txt(shifted_txt)