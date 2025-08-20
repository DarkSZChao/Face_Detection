import glob
import itertools
import os
import random
import shutil

from tools.dataset_reader import get_trial_dir_list

# set target dataset dir
train_img_dir = './dataset/train/images'
train_label_dir = './dataset/train/labels'
val_img_dir = './dataset/val/images'
val_label_dir = './dataset/val/labels'
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# get data
img_path_list = []
img_dir_list = get_trial_dir_list('D:/dataset_pending', (0, 900))
for f in img_dir_list:
    # for the subfolder with the most of the faces in each folder
    img_subfolder_No_list = [int(os.path.basename(f).split('_')[0]) for f in glob.glob(f'{f}/[!no]_face_detected')]
    max_No = max(img_subfolder_No_list)
    img_path_list.append(glob.glob(f'{f}/{max_No}_face_detected/*.png'))
img_path_list = list(itertools.chain(*img_path_list))

train_img_paths = random.sample(img_path_list, int(len(img_path_list) * 0.9))
train_label_paths = [os.path.join(os.path.dirname(i), os.path.basename(i).split('.')[0] + '.txt') for i in train_img_paths]
val_img_paths = list(set(img_path_list) - set(train_img_paths))
val_label_paths = [os.path.join(os.path.dirname(i), os.path.basename(i).split('.')[0] + '.txt') for i in val_img_paths]

for idx, (img, label) in enumerate(zip(train_img_paths, train_label_paths)):
    print(f'Working: [{idx}/{len(train_img_paths)}]')
    shutil.copy2(img, train_img_dir)
    shutil.copy2(label, train_label_dir)
for idx, (img, label) in enumerate(zip(val_img_paths, val_label_paths)):
    print(f'Working: [{idx}/{len(val_img_paths)}]')
    shutil.copy2(img, val_img_dir)
    shutil.copy2(label, val_label_dir)

print("Dataset for YOLO is ready!")
