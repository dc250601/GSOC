import os
import random
import shutil


def curzon(parameter):
    VALIDATION_SPLIT = 0.2
    path_0 = "./Test/0"
    path_1 = "./Test/1"
    os.makedirs(f"./Data_small_{parameter}/Train/0")
    os.makedirs(f"./Data_small_{parameter}/Train/1")
    os.makedirs(f"./Data_small_{parameter}/Test/0")
    os.makedirs(f"./Data_small_{parameter}/Test/1")

    list_image_0 = os.listdir(path_0)
    random.shuffle(list_image_0)
    british(list_image_0, "./Test/0/", f"./Data_small_{parameter}/Train/0/", f"./Data_small_{parameter}/Test/0/", VALIDATION_SPLIT)

    list_image_1 = os.listdir(path_1)
    random.shuffle(list_image_1)
    british(list_image_1, "./Test/1/", f"./Data_small_{parameter}/Train/1/", f"./Data_small_{parameter}/Test/1/", VALIDATION_SPLIT)


def british(lst, path, new_path_train, new_path_test, VALIDATION_SPLIT):
    train_length = int(len(lst)*(1-VALIDATION_SPLIT))
    for image in lst[:train_length]:
        ori_path = path+str(image)
        shutil.move(ori_path, (new_path_train+str(image)))
    for image in lst[train_length:]:
        ori_path = path+str(image)
        shutil.move(ori_path, (new_path_test+str(image)))