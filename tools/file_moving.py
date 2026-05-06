import glob
import os.path
import shutil
from multiprocessing import Pool


def process_subfolder(idx, total, input_sub_dir, output_dir):
    print(f"[{idx + 1}/{total}] Working on: [{input_sub_dir}] to [{output_dir}]")

    for item in glob.glob(f'{input_sub_dir}/**/*.*', recursive=True):
        dst = os.path.join(input_sub_dir, os.path.basename(item))
        if os.path.abspath(item) == os.path.abspath(dst):
            continue

        shutil.move(item, input_sub_dir)

    # delete empty folders
    empty_dir = [d for d in glob.glob(f'{input_sub_dir}/*') if os.path.isdir(d)]
    for folder in empty_dir:
        if not os.listdir(folder):
            os.rmdir(folder)


if __name__ == "__main__":
    input_dir = r'.\marked_results'
    output_dir = r'.\marked_results'

    input_sub_dir_list = [d for d in glob.glob(f'{input_dir}/*') if os.path.isdir(d)]

    args_list = [
        (i, len(input_sub_dir_list), input_sub_dir, output_dir)
        for i, input_sub_dir in enumerate(input_sub_dir_list)
    ]

    with Pool(processes=1) as pool:
        pool.starmap(process_subfolder, args_list, chunksize=1)
