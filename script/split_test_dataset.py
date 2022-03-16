import os
import pathlib


def split_to_test_dataset(src_dir, target_dir):
    for class_dir in os.listdir(src_dir):
        files = os.listdir(pathlib.Path(src_dir, class_dir))[-50:]
        print(files)
        if len(files) < 440:
            print("file number is too small")
            continue
        for file in files:
            old_name = pathlib.Path(src_dir, class_dir, file)
            new_name = pathlib.Path(target_dir, class_dir, file)
            os.renames(old_name, new_name)

if __name__ == '__main__':
    split_to_test_dataset("../Data/Data", "../Data/Data_test")