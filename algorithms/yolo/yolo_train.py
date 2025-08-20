import torch
from ultralytics import YOLO


def train_yolov8():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="./dataset/data.yaml",
        device='0',
        epochs=30,
        batch=16,
        imgsz=960,  # input resolution
        lr0=0.001,  # initial learning rate
        patience=0,  # disable early-stop
        augment=False,  # disable data augmentation
        overlap_mask=False,
        weight_decay=0,
        single_cls=True,  # single class
        optimizer="SGD",
        name="yolov8s_face_finetuned"  # output model name
    )


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())  # True means GPU is available
    print(torch.cuda.get_device_name(0))

    train_yolov8()

    # predict_and_blur(
    #     image_path="test.jpg",
    #     output_dir="blurred_results"
    # )
