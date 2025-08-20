import glob
import os

import cv2
from ultralytics import YOLO


class S_yolo:
    def __init__(self):
        self.model = YOLO(glob.glob("**/runs_best/detect/yolov8s_face_finetuned/weights/best.pt", recursive=True)[0])  # load yolo

    def process(self, input_path):
        img = cv2.imread(input_path)

        # apply detection method
        boxes = self.model.predict(img, conf=0.01, iou=0.1, verbose=False)[0].boxes.xyxy  # low threshold

        boxes_list = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # make sure no exceed the image boundary
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # convert to standard format
            center_x = format(float((x2 + x1) / (2 * img.shape[1])), ".6f")
            center_y = format(float((y2 + y1) / (2 * img.shape[0])), ".6f")
            width = format(float((x2 - x1) / img.shape[1]), ".6f")
            height = format(float((y2 - y1) / img.shape[0]), ".6f")
            box_normalised = (center_x, center_y, width, height)
            boxes_list.append(box_normalised)
        return img, boxes_list


def multiple_img_predict(input_dir, output_dir, save_img=False):
    os.makedirs(output_dir, exist_ok=True)

    input_path_list = glob.glob(input_dir + f'/*.png')
    for input_path in input_path_list:
        img, box_list = S_yolo().process(input_path)

        # save results
        if save_img:
            cv2.imwrite(output_dir + '/' + os.path.basename(input_path), img)

        with open(output_dir + '/' + os.path.basename(input_path).split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
            for box in box_list:
                line = f'0 {box[0]} {box[1]} {box[2]} {box[3]}'
                f.write(line + '\n')

    print(f"Prediction saved to: {output_dir}")


if __name__ == "__main__":
    # img_dir_list = get_trial_dir_list('./dataset_pending', (0, 900))
    img_dir_list = glob.glob('../test/*')

    # for each folder
    for f in img_dir_list:
        multiple_img_predict(f, f'../test_results', save_img=False)
