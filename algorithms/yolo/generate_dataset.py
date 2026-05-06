import glob
import itertools
import os
import random
import shutil
from multiprocessing import Pool


def copy_pair(img_path, label_path, input_dir, img_out_dir, label_out_dir):
    tag = '_'.join(os.path.split(os.path.dirname(os.path.relpath(img_path, input_dir))))
    new_img_name = f"{tag}_{os.path.basename(img_path)}"
    new_label_name = f"{tag}_{os.path.basename(label_path)}"

    shutil.copy2(img_path, os.path.join(img_out_dir, new_img_name))
    shutil.copy2(label_path, os.path.join(label_out_dir, new_label_name))


if __name__ == "__main__":
    # set output dataset dir
    train_img_output_dir = './dataset/train/images'
    train_label_output_dir = './dataset/train/labels'
    val_img_output_dir = './dataset/val/images'
    val_label_output_dir = './dataset/val/labels'
    os.makedirs(train_img_output_dir, exist_ok=True)
    os.makedirs(train_label_output_dir, exist_ok=True)
    os.makedirs(val_img_output_dir, exist_ok=True)
    os.makedirs(val_label_output_dir, exist_ok=True)

    # set input dataset dir
    input_dir = r'.\good_results'
    input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/*') if os.path.isdir(d)]

    # get data
    img_path_list = []
    for f in input_sub_dir_list:
        # for the subfolder with the most of the faces in each folder
        # max_No = max([int(os.path.basename(f).split('_')[0]) for f in glob.glob(f'{f}/[!no]_face_detected')])
        # max_No = 1
        # img_path_list.append(glob.glob(f'{f}/{max_No}_face_detected/*.png'))
        img_path_list.append(glob.glob(f'{f}/**/*.png', recursive=True))
    img_path_list = list(itertools.chain(*img_path_list))

    train_img_paths = random.sample(img_path_list, int(len(img_path_list) * 0.95))
    train_label_paths = [os.path.join(os.path.dirname(i), os.path.basename(i).split('.')[0] + '.txt') for i in train_img_paths]
    val_img_paths = list(set(img_path_list) - set(train_img_paths))
    val_label_paths = [os.path.join(os.path.dirname(i), os.path.basename(i).split('.')[0] + '.txt') for i in val_img_paths]

    # train
    train_args = list(zip(
        train_img_paths,
        train_label_paths,
        [input_dir] * len(train_img_paths),
        [train_img_output_dir] * len(train_img_paths),
        [train_label_output_dir] * len(train_img_paths),
    ))
    with Pool(processes=20) as pool:
        list(pool.starmap(copy_pair, train_args))

    # val
    val_args = list(zip(
        val_img_paths,
        val_label_paths,
        [input_dir] * len(val_img_paths),
        [val_img_output_dir] * len(val_img_paths),
        [val_label_output_dir] * len(val_img_paths),
    ))
    with Pool(processes=20) as pool:
        list(pool.starmap(copy_pair, val_args))

    print("Dataset for YOLO is ready!")
