import cv2
from insightface.app import FaceAnalysis


class A_retinaface:
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=['detection'])  # apply RetinaFace
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def process(self, input_path):
        img = cv2.imread(input_path)

        # apply detection method
        faces = self.app.get(img)

        boxes_list = []
        conf_list = []
        for face in faces:
            box = face.bbox.astype(int)

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

            conf = format(float(face.det_score), ".6f")

            boxes_list.append((center_x, center_y, width, height))
            conf_list.append(conf)

        return img, boxes_list, conf_list


if __name__ == "__main__":
    input_path = "../1/1_0_233.png"

    handler = A_retinaface()
    img, boxes_list, _ = handler.process(input_path)
    pass
