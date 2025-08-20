import glob
import os

from tools.dataset_reader import get_trial_dir_list

if __name__ == "__main__":
    img_dir_list1 = get_trial_dir_list('C:/Users/Desktop/extracted_images', (0, 900))
    img_dir_list2 = get_trial_dir_list('D:/blurred_images', (0, 900))

    # for each folder
    for dir1, dir2 in zip(img_dir_list1, img_dir_list2):
        png1 = glob.glob(os.path.join(dir1, '*.png'))
        png2 = glob.glob(os.path.join(dir2, '*.png'))

        if len(png1) == len(png2):
            # print('Match', len(png1), len(png2))
            pass
        else:
            print('Mismatch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(png1), len(png2), dir1, dir2)
