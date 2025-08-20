import cv2
from mtcnn import MTCNN


class B_mtcnn:
    def __init__(self):
        self.model = MTCNN()

    def process(self, input_path):
        img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)

        # apply detection method
        boxes = self.model.detect_faces(img)

        boxes_list = []
        for box in boxes:
            x, y, w, h = box['box']
            # make sure no exceed the image boundary
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            # convert to standard format
            center_x = format(float((x + w / 2) / img.shape[1]), ".6f")
            center_y = format(float((y + h / 2) / img.shape[0]), ".6f")
            width = format(float(w / img.shape[1]), ".6f")
            height = format(float(h / img.shape[0]), ".6f")
            box_normalised = (center_x, center_y, width, height)
            boxes_list.append(box_normalised)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), boxes_list


if __name__ == "__main__":
    input_path = "../1/1_0_233.png"

    handler = B_mtcnn()
    img, boxes_list = handler.process(input_path)
    pass
