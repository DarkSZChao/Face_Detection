# Face Detection & Blurring for Data Anonymization 

This repository provides a tool to **detect faces in images** and save both the processed images and corresponding bounding box annotations. It supports multiple face detection algorithms and can optionally split results based on the number
of detected faces.

---

## ✨ Features

- Detect faces using multiple algorithms:
    - `S_yolo` (custom, trainable)
    - `A_retinaface` (open-source)
    - `B_mtcnn` (open-source)
    - `D_opencv_caffe` (open-source)
    - `D_face_recognition` (open-source)
- The prefix letter indicates the **performance level** of the algorithm, Best to Worst: `S`→`A`→`B`→`C`→`D`
- Generate `.txt` label files with detected face coordinates following YOLO box format: (id, center_x, center_y, width, height) normalised 0~1.

---

## 📂 Project Structure

```
.
├── algorithms/                   # Implementations of different face detection algorithms
│   └── opencv_caffe              # Model files for opencv caffe
│       └── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
│   └── yolo                      # Customed YOLO model
│       └── dataset               # Dataset to train YOLO
│       └── runs_best             # Best run with YOLO weights
│       └── dataset_generator.py  # Generate dataset for YOLO training
│       └── yolo_train.py         # YOLO training script
│       └── yolo11n.pt            # Pretrained YOLO model
│       └── yolov8s.pt            # Pretrained YOLO model
│   └── S_yolo.py                 # S-class YOLO algorithm
│   └── A_retinaface.py           # A-class algorithm
│   └── B_mtcnn.py                # B-class algorithm
│   └── D_face_recognition.py     # D-class algorithm
│   └── D_opencv_caffe.py         # D-class algorithm
├── tools/                        # Utility scripts
│   └── image_bug                 # Detect corrupted images
│   └── check_result_matching.py  # Check whether the number of output images matches the input
│   └── dataset_reader.py         # Load subfoldera in the dataset for specific structure
│   └── file_moving.py            # Move files
├── apply_blur.py                 # Script to apply blur effect on images based on the box coordinates
├── dataset_viewer.py             # Script to visualize images and draw boxex
├── main.py                       # Main entry point to run face detection & labeling
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore file
```

