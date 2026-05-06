import glob
import os
import shutil
from time import sleep

import cv2


def should_skip(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            _, xc, yc, bw, bh = map(float, line.strip().split())

            if 0.99 < xc < 1 and 0.12 < yc < 0.2:
                return True
            if 0.54 < xc < 0.56 and 0.54 < yc < 0.56:
                return True
    return False


def visualize_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, xc, yc, bw, bh = map(float, line.strip().split())
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Image Slideshow', img)


if __name__ == "__main__":
    autu_play = False

    input_dir = r'.\marked_results'
    output_dir = r'.\good_results'

    # input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/IA[0-9]-P[0-9]-*') if os.path.isdir(d)][0:]
    # input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/IA[0-9]-P[0-9]P[0-9]-*') if os.path.isdir(d)][0:]
    input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/M*') if os.path.isdir(d)][0:]
    # for each folder
    for f in input_sub_dir_list:
        print(f'Displaying folder: {f}')

        skip_folder = False

        img_path_list = glob.glob(f'{f}/2_face_detected/**/*.png', recursive=True)
        # sort the images
        img_path_list = sorted(
            img_path_list,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        # for each image
        for img_path in img_path_list:
            print(f'Image: {os.path.basename(img_path)}')
            label_path = f'{os.path.dirname(img_path)}/{os.path.basename(img_path).split('.')[0]}.txt'

            # custom filter: skip if there is a box in the unwanted area
            if should_skip(label_path):
                print(f"Skipped (top-right): {img_path}")
                continue

            visualize_boxes(img_path=img_path, label_path=label_path)

            if autu_play:
                key = cv2.waitKey(500) & 0xFF  # ms
                if key == 13:  # enter key to skip
                    pass
                elif key == 32:  # space key to pause
                    while 1:
                        if cv2.waitKey(200) & 0xFF == 32:
                            break
                elif key == 27:  # ESC key to exit
                    raise Exception
            else:
                while True:
                    key = cv2.waitKey(20) & 0xFF  # ms
                    if key == ord('0'):
                        dst_img_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
                        dst_label_path = os.path.join(output_dir, os.path.relpath(label_path, input_dir))

                        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                        shutil.copy(img_path, dst_img_path)
                        shutil.copy(label_path, dst_label_path)
                        print(f"Copied label to: {dst_label_path}, image to: {dst_img_path}")
                        break
                    elif key == 13:  # enter key to skip this image
                        break
                    elif key == ord('.'):  # to skip the folder
                        skip_folder = True
                        break
                    elif key == 27:  # ESC key to exit
                        raise Exception

            if skip_folder:
                break
        if skip_folder:
            continue

        sleep(0.5)
    cv2.destroyAllWindows()
