import glob
import os

import cv2
from ultralytics import YOLO


class S_yolo:
    def __init__(self, model=None):
        if model is None:
            self.model = YOLO(glob.glob("**/runs_best/detect/2026-05-05_18-08_yolov8s_face_finetuned/weights/best.pt", recursive=True)[0])  # load yolo
            # self.model = YOLO(glob.glob("**/yolov8s.pt", recursive=True)[0])  # load yolo

    def process(self, input_path):
        img = cv2.imread(input_path)

        # apply detection method
        result = self.model.predict(img, conf=0.01, iou=0.1, verbose=False)[0]  # low threshold
        boxes = result.boxes.xyxy
        confs = result.boxes.conf

        boxes_list = []
        conf_list = []
        for box, conf in zip(boxes, confs):
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

            conf = format(float(conf), ".6f")

            boxes_list.append((center_x, center_y, width, height))
            conf_list.append(conf)

        return img, boxes_list, conf_list


def multiple_img_predict(input_dir, output_dir, save_img=False):
    os.makedirs(output_dir, exist_ok=True)

    input_path_list = glob.glob(input_dir + f'/*.png')
    for input_path in input_path_list:
        img, box_list, _ = S_yolo().process(input_path)

        # save results
        if save_img:
            cv2.imwrite(output_dir + '/' + os.path.basename(input_path), img)

        with open(output_dir + '/' + os.path.basename(input_path).split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
            for box in box_list:
                line = f'0 {box[0]} {box[1]} {box[2]} {box[3]}'
                f.write(line + '\n')

    print(f"Prediction saved to: {output_dir}")


if __name__ == "__main__":
    img_dir_list = glob.glob('../test/*')

    # for each folder
    for f in img_dir_list:
        multiple_img_predict(f, f'../test_results', save_img=False)
