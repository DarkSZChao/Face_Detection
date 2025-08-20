import glob
import re

# for the dataset structure
def get_trial_dir_list(input_dir, idx=(0, 900)):
    img_dir_list = glob.glob(input_dir + '/*')
    img_dir_list = [
        folder for folder in img_dir_list
        if idx[0] <= int(re.split(f'[{re.escape('/')}{re.escape('\\')}]', folder)[-1]) <= idx[1]
    ]
    img_dir_list = sorted(img_dir_list, key=lambda x: int(x.split('\\')[-1]))  # sort the folder

    return img_dir_list


if __name__ == "__main__":
    img_dir_list = get_trial_dir_list('D:/dataset_pending', (0, 480))
    pass