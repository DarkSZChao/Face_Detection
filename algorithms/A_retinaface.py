import cv2
from insightface.app import FaceAnalysis


class A_retinaface:
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=['detection'])  # apply RetinaFace
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def process(self, input_path):
        img = cv2.imread(input_path)

        # apply detection method
        boxes = self.app.get(img)

        boxes_list = []
        for box in boxes:
            box = box.bbox.astype(int)
            # make sure no exceed the image boundary
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img.shape[1], box[2])
            y2 = min(img.shape[0], box[3])

            # convert to standard format
            center_x = format(float((x2 + x1) / (2 * img.shape[1])), ".6f")
            center_y = format(float((y2 + y1) / (2 * img.shape[0])), ".6f")
            width = format(float((x2 - x1) / img.shape[1]), ".6f")
            height = format(float((y2 - y1) / img.shape[0]), ".6f")
            box_normalised = (center_x, center_y, width, height)
            boxes_list.append(box_normalised)
        return img, boxes_list


if __name__ == "__main__":
    input_path = "../1/1_0_233.png"

    handler = A_retinaface()
    img, boxes_list = handler.process(input_path)
    pass
