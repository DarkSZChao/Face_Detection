import glob

import cv2
import numpy as np


class D_opencv_caffe:
    def __init__(self):
        model = glob.glob("./**/opencv_caffe/res10_300x300_ssd_iter_140000.caffemodel", recursive=True)[0]
        config = glob.glob("./**/opencv_caffe/deploy.prototxt", recursive=True)[0]
        self.model = cv2.dnn.readNetFromCaffe(config, model)

    def process(self, input_path):
        img = cv2.imread(input_path)

        # apply detection method
        self.model.setInput(cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False))
        detections = self.model.forward()

        boxes_list = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                x1, y1, x2, y2 = box.astype(int)
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


if __name__ == "__main__":
    input_path = "../test/1/1_0_233.png"

    handler = D_opencv_caffe()
    img, boxes_list = handler.process(input_path)
    pass
