import os
import sys
import imghdr

import PIL
import numpy as np
from environs import Env
from PIL import Image

env = Env()
env.read_env("./../.env")

def check(dir_personal):
    """
    check if all photos' size
    """
    with open("illegal.txt", "a+") as f:
        for photo in os.listdir(dir_personal):
            path = os.path.join(dir_personal, photo)
            try:
                photo_format = imghdr.what(path)
                new_name = path.split(".")[0] + os.path.basename(dir_personal) + "_" + "." + photo_format
                os.renames(path, new_name)
                path = new_name
                img = Image.open(path)
                if img.size != (200, 200):
                    print(path)
                    os.remove(path)
                    f.write(f"path: {os.path.basename(dir_personal)}/{os.path.basename(path)} the size({img.size}) is not correct\n")
            except PIL.UnidentifiedImageError as e:
                print(e)
                os.remove(path)
                f.write(f"path: {os.path.basename(dir_personal)}/{os.path.basename(path)}  error image \n")


def check_dirs(dir_all):
    """
    check sub-directory's photos' size
    """
    for item in os.listdir(dir_all):
        dir_personal = os.path.join(dir_all, item)
        if not os.path.isdir(dir_personal):
            continue
        check(dir_personal)


def cp_to_dataset(dir_all):
    """
    mv to dataset for training
    """
    data_root = env.str("data_root")
    for item in os.listdir(dir_all):
        dir_personal = os.path.join(dir_all, item)
        if not os.path.isdir(dir_personal):
            continue
        dir_personal += "/*"
        dst = "../" + data_root + "/water"
        if not os.path.exists(dst):
            os.makedirs(dst)
        os.system(f"cp {dir_personal} {dst}")


def rename_dataset(path=None):
    """
    rename to compoly teacher's rule
    """
    if path is None:
        path = "../" + os.getenv("data_root") + "/water"
    files = np.array(os.listdir(path))
    np.random.shuffle(files)
    for i in range(files.size):
        old_name = os.path.join(path, files[i])
        new_name = os.path.join(path, "d" + add_0_before_int(i) + ".jpg")
        os.renames(old_name, new_name)


def add_0_before_int(x: int):
    """
    add zero before a str to compoly the length
    """
    x = str(x)
    while len(x) < 3:
        x = "0" + x
    return x


def convert_to_jpg(path=None):
    """
    convert png to jpg
    """
    if path is None:
        path = "../" + os.getenv("data_root") + "/water"
    files = np.array(os.listdir(path))
    for file in files:
        if str(file).endswith("jpeg"):
            os.renames(os.path.join(path, file), os.path.join(path, file.split(".")[0] + ".jpg"))
            continue
        if  str(file).endswith("jpg"):
            continue
        print(file)
        im = PIL.Image.open(os.path.join(path, file))
        os.remove(os.path.join(path, file))
        im.save(os.path.join(path, file.split(".")[0] + ".jpg"))




if __name__ == '__main__':
    check_dirs("/media/andy/z/python/毕业实习/制作数据集")
    cp_to_dataset("/media/andy/z/python/毕业实习/制作数据集")
    convert_to_jpg()
    rename_dataset()
