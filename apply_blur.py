import glob
import os

import cv2

from tools.dataset_reader import get_trial_dir_list


def img_blur(img_path, label_path, output_dir):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        # print(label_path)
        for line in f.readlines():
            # if line.strip() == '':
            #     continue
            class_id, xc, yc, bw, bh = map(float, line.strip().split())
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (99, 99), 30)

    output_dir = output_dir + f'/{os.path.dirname(img_path).split('\\')[-1]}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = str(os.path.join(output_dir, os.path.basename(img_path)))
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    img_dir_list = get_trial_dir_list('./test_results', (0, 900))

    # for each folder
    for f in img_dir_list:
        print(f'Working on folder: {f}')

        img_path_list = glob.glob(f'{f}/*.png')[0:]

        # for each image
        for img_path in img_path_list:
            label_path = f'{os.path.dirname(img_path)}/{os.path.basename(img_path).split('.')[0]}.txt'
            img_blur(img_path=img_path,
                     label_path=label_path,
                     output_dir='./test_results/blurred_images',
                     )


