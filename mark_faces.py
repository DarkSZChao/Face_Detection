# file_name is os.path.basename()
# file_dir is os.path.dirname()
# file_path is dir+name

import csv
import glob
import os
from multiprocessing import Pool, Process, Queue

import cv2

# =========================
# GLOBAL
# =========================
handler = None
q = None


def init_worker(algorithm, queue):
    global handler, q
    q = queue

    if algorithm == 'S_yolo':
        from algorithms.S_yolo import S_yolo
        handler = S_yolo()

    elif algorithm == 'A_retinaface':
        from algorithms.A_retinaface import A_retinaface
        handler = A_retinaface()

    elif algorithm == 'B_mtcnn':
        from algorithms.B_mtcnn import B_mtcnn
        handler = B_mtcnn()

    elif algorithm == 'D_opencv_caffe':
        from algorithms.D_opencv_caffe import D_opencv_caffe
        handler = D_opencv_caffe()

    elif algorithm == 'D_face_recognition':
        from algorithms.D_face_recognition import D_face_recognition
        handler = D_face_recognition()

    else:
        raise ValueError(f'Algorithm {algorithm} is not found!')

    print(f"[Worker Init] Loaded model: {algorithm}")


def image_faces_label(idx, total, input_sub_dir, output_dir, result_split=False):
    """
    Find faces in all images and label them

    Param:
        idx: the index of this job
        total: the total number of jobs
        input_sub_dir: input image dir
        output_dir: output dir
        algorithm: the algorithm applied
        result_split: split the result based on the number of faces
    """
    global handler, q

    print(f"[{idx + 1}/{total}] Working on: [{input_sub_dir}] to [{output_dir}]")
    input_dir = os.path.dirname(input_sub_dir)

    # find all images in the input_dir
    for img_path in glob.glob(f"{input_sub_dir}/**/*.png", recursive=True):
        img, face_boxes, confidences = handler.process(img_path)  # process this image and locate the faces

        img_relpath = os.path.relpath(img_path, input_dir)  # keep the input folder structure
        img_reldir = os.path.dirname(img_relpath)  # keep the input folder structure
        img_basename = os.path.basename(img_relpath)  # keep the input folder structure
        txt_basename = '.'.join(img_basename.split('.')[:-1])
        face_No = len(face_boxes)

        # split the results based on face detection result
        if result_split:
            output_sub_dir = os.path.join(output_dir, img_reldir, f"{face_No}_face_detected")
        else:  # no split
            output_sub_dir = os.path.join(output_dir, img_reldir)

        # create output sub dir
        os.makedirs(output_sub_dir, exist_ok=True)
        output_img_path = os.path.join(output_sub_dir, img_basename)
        output_label_path = os.path.join(output_sub_dir, f'{txt_basename}.txt')

        # save the image
        cv2.imwrite(output_img_path, img)  # keep the input folder structure

        # save YOLO box info
        with open(output_label_path, 'w', encoding='utf-8') as f:
            for face_box in face_boxes:
                line = f'0 {face_box[0]} {face_box[1]} {face_box[2]} {face_box[3]}'  # class_id is 0 for face
                f.write(line + '\n')

        # put info into queue for csv writing
        relpath_for_csv = os.path.relpath(output_img_path, output_dir)
        if face_No == 0:
            q.put([relpath_for_csv, -1, None, None, None, None, None])
        else:
            for face_box, conf in zip(face_boxes, confidences):
                q.put([
                    relpath_for_csv,
                    0,  # class_id is 0 for face
                    float(face_box[0]),
                    float(face_box[1]),
                    float(face_box[2]),
                    float(face_box[3]),
                    float(conf)
                ])

        # print(f"{face_No} faces detected:\tImage: {output_img_path}\tLabel: {output_label_path}")


def writer(queue, output_csv):
    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            "relpath",
            "face_id",
            "x_center",
            "y_center",
            "width",
            "height",
            "confidence"
        ])

        count = 0
        while True:
            item = queue.get()
            if item == "DONE":
                break
            w.writerow(item)
            count += 1
            if count % 10000 == 0:
                print(f"[Writer] rows written: {count}")


if __name__ == "__main__":
    multiprocessing_enable = True
    result_split = False
    algorithm = 'S_yolo'

    input_dir = r'.\extracted_images'
    output_dir = r'.\marked_results'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "summary.csv")

    input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/*') if os.path.isdir(d)]

    args_list = [
        (i, len(input_sub_dir_list), input_sub_dir, output_dir, result_split)
        for i, input_sub_dir in enumerate(input_sub_dir_list)
    ]

    queue = Queue(maxsize=20000)
    writer_p = Process(target=writer, args=(queue, output_csv))
    writer_p.start()

    # use multiprocessing to process images in parallel
    if multiprocessing_enable:
        with Pool(processes=14, initializer=init_worker, initargs=(algorithm, queue)) as pool:
            pool.starmap(image_faces_label, args_list, chunksize=1)
    else:
        init_worker(algorithm, queue)
        for args in args_list:
            image_faces_label(*args)

    # signal the writer process to finish
    queue.put("DONE")
    writer_p.join()
    print(f"[DONE] CSV saved at: {output_csv}")
