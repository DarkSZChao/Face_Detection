import glob
import os.path
import shutil

from send2trash import send2trash

from tools.dataset_reader import get_trial_dir_list

if __name__ == "__main__":
    img_dir_list = get_trial_dir_list('./dataset_pending', (0, 840))

    for f in img_dir_list:
        print(f'Working: [{f}]')

        everything_path_list = glob.glob(f'{f}/*_face_detected/*')
        for item in everything_path_list:
            shutil.copy2(item, f)

        # everything_path_list = glob.glob(f'{f}/[!0!3]_face_detected/*')
        # os.makedirs(f + '/0_face_detected', exist_ok=True)
        # for item in everything_path_list:
        #     shutil.move(item, f + '/0_face_detected')
        #
        # send2trash(f + '/1_face_detected')
        # send2trash(f + '/2_face_detected')
