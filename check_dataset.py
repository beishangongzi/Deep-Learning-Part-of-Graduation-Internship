import os
import pathlib

from PIL import Image

def get_the_first_file_of_dir(path):
    if os.path.isdir(path):
        return get_the_first_file_of_dir(os.path.join(path, os.listdir(path)[0]))
    return os.path.abspath(path)

def check_dataset(path):
    """
    check numer of each class
    check the size of each class
    """
    class_names = os.listdir(path)
    num_class = len(class_names)
    num_each_class = []
    for class_name in class_names:
        num_each_class.append({class_name: len(os.listdir(os.path.join(path, class_name)))})

    picture_size = Image.open(get_the_first_file_of_dir(path)).size
    print(f"""class_name: {class_names}
number of class: {num_class}
num_each_class: {num_each_class}
picture_size: {picture_size}""")

if __name__ == '__main__':
    check_dataset("Data/Data2/flower_photos")