import glob
import os
from time import sleep

import cv2

from tools.dataset_reader import get_trial_dir_list


def visualize_labels(img_path, label_path, output_dir=None):
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

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = str(os.path.join(output_dir, os.path.basename(img_path)))
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    img_dir_list = get_trial_dir_list('./test_results', (0, 900))

    # for each folder
    for f in img_dir_list:
        print(f'Displaying folder: {f}')

        # # for the subfolder with the most of the faces in each folder
        # img_subfolder_No_list = [int(os.path.basename(f).split('_')[0]) for f in glob.glob(f'{f}/*_face_detected')]
        # No = 0
        # img_path_list = glob.glob(f'{f}/{No}_face_detected/*.png')[0:]
        img_path_list = glob.glob(f'{f}/*.png')[0:]

        # for each image
        for img_path in img_path_list:
            print(f'Image: {os.path.basename(img_path)}')
            label_path = f'{os.path.dirname(img_path)}/{os.path.basename(img_path).split('.')[0]}.txt'
            visualize_labels(img_path=img_path,
                             label_path=label_path,
                             # output_dir='./test',
                             )
            key = cv2.waitKey(500) & 0xFF  # ms
            if key == 13:  # enter key to skip
                pass
            elif key == 32:  # space key to pause
                while 1:
                    if cv2.waitKey(200) & 0xFF == 32:
                        break
            elif key == 27:  #ESC key to exit
                raise Exception

        sleep(0.5)
    cv2.destroyAllWindows()