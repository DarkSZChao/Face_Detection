# file_name is os.path.basename()
# file_dir is os.path.dirname()
# file_path is dir+name

import glob
import os
from multiprocessing import Pool

import cv2

from tools.dataset_reader import get_trial_dir_list


def image_faces_label(input_dir, output_dir, algorithm='A_retinaface', result_split=False):
    """
    Find faces in all images and label them

    Param:
        input_dir: input image dir
        output_dir: output dir
        algorithm: the algorithm applied
        result_split: split the result based on the number of faces
    """
    os.makedirs(output_dir, exist_ok=True)

    # select algorithm
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
        raise ValueError(f'Algorithm {[algorithm]} is not found!')

    # find all images in the input_dir
    for img_path in glob.glob(f"{input_dir}/**/*.png", recursive=True):
        img, face_boxes = handler.process(img_path)  # process this image and locate the faces

        img_subpath = os.path.relpath(img_path, input_dir)  # keep the input folder structure
        img_subdir = os.path.dirname(img_subpath)  # keep the input folder structure
        img_basename = os.path.basename(img_subpath)  # keep the input folder structure
        face_No = len(face_boxes)

        # split the results based on face detection result
        if result_split:
            output_dir_split = os.path.join(output_dir, f"{face_No}_face_detected")
            output_path = os.path.join(output_dir_split, img_basename)
            os.makedirs(output_dir_split, exist_ok=True)
            cv2.imwrite(output_path, img)  # ignore the input folder structure
            print(f"{face_No} face detected:\t{img_path} -> {output_path}")

            # save box info
            if face_No > 0:
                txt_basename = '.'.join(img_basename.split('.')[:-1])
                with open(os.path.join(output_dir_split, f'{txt_basename}.txt'), 'w', encoding='utf-8') as f:
                    for face_box in face_boxes:
                        line = f'0 {face_box[0]} {face_box[1]} {face_box[2]} {face_box[3]}'
                        f.write(line + '\n')

        else:  # no split
            output_dir_nosplit = os.path.join(output_dir, img_subdir)
            output_path = os.path.join(output_dir_nosplit, img_basename)
            os.makedirs(output_dir_nosplit, exist_ok=True)
            cv2.imwrite(output_path, img)  # keep the input folder structure
            print(f"{face_No} faces detected:\t{img_path} -> {output_path}")

            # save box info
            txt_basename = '.'.join(img_basename.split('.')[:-1])
            with open(os.path.join(output_dir_nosplit, f'{txt_basename}.txt'), 'w', encoding='utf-8') as f:
                for face_box in face_boxes:
                    line = f'0 {face_box[0]} {face_box[1]} {face_box[2]} {face_box[3]}'
                    f.write(line + '\n')


if __name__ == "__main__":
    multiprocessing_enable = True

    # set dir
    input_dir = './test'
    output_dir = './test_results'
    # get subfolder and allocate the job at subfolder level
    input_subfolder_list = get_trial_dir_list(input_dir, (0, 900))
    output_subfolder_list = [os.path.join(output_dir, os.path.basename(folder)) for folder in input_subfolder_list]
    # generate args list for each subfolder
    args_list = [(input_dir, output_dir, 'D_face_recognition', False) for input_dir, output_dir in zip(input_subfolder_list, output_subfolder_list)]

    if multiprocessing_enable:
        with Pool(processes=4) as pool:
            pool.starmap(image_faces_label, args_list)
    else:
        for args in args_list:
            image_faces_label(*args)
