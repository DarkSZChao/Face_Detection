import glob
import os
from multiprocessing import Pool


def check_matching(idx, total, dir1, dir2):
    print(f"[{idx + 1}/{total}] Working on: [{dir1}] and [{dir2}]")

    items1 = set(os.path.basename(d) for d in glob.glob(f'{dir1}/*.png'))
    items2 = set(os.path.basename(d) for d in glob.glob(f'{dir2}/*.png'))

    only_in_dir1 = items1 - items2
    only_in_dir2 = items2 - items1

    if only_in_dir1:
        print(f'❌ Only in {dir1}: {only_in_dir1}')
    # else:
        # print(f'✅ All items in {dir1} are present in {dir2}')

    if only_in_dir2:
        print(f'❌ Only in {dir2}: {only_in_dir2}')
    # else:
        # print(f'✅ All items in {dir2} are present in {dir1}')

if __name__ == "__main__":
    input_dir1 = r'.\extracted_images'
    input_sub_dir_list1 = [d for d in glob.glob(f'{input_dir1}/*') if os.path.isdir(d)]
    input_dir2 = r'.\marked_results'
    input_sub_dir_list2 = [d for d in glob.glob(f'{input_dir2}/*') if os.path.isdir(d)]

    args_list = [
        (i, len(input_sub_dir_list1), input_sub_dir1, input_sub_dir2)
        for i, (input_sub_dir1, input_sub_dir2) in enumerate(zip(input_sub_dir_list1, input_sub_dir_list2))
    ]

    with Pool(processes=20) as pool:
        pool.starmap(check_matching, args_list, chunksize=1)
