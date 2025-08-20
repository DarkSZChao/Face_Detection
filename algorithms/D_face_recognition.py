import cv2
import face_recognition


class D_face_recognition:
    def __init__(self):
        self.model = face_recognition

    def process(self, input_path):
        img = self.model.load_image_file(input_path)

        # apply detection method
        boxes = self.model.face_locations(img)

        boxes_list = []
        for box in boxes:
            y1, x2, y2, x1 = box
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
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), boxes_list


if __name__ == "__main__":
    input_path = "../1/1_0_233.png"

    handler = D_face_recognition()
    img, boxes_list = handler.process(input_path)
    pass
