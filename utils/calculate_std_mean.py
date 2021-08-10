import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageStat
import numpy as np
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import os
import sys
import time


def calculate_std_mean_from_imgs(src_data_folder="../dataset/images/train2021", target_dir=''):
    img_all_data = os.listdir(src_data_folder)
    cnt = 0
    fst_moment = np.zeros(3)  # torch.empty(3)
    snd_moment = np.zeros(3)  # torch.empty(3)
    print(len(img_all_data))
    for i in tqdm(range(len(img_all_data))):
        img_path = os.path.join(src_data_folder, img_all_data[i])
        img = Image.open(img_path)
        stat = ImageStat.Stat(img)
        image_width, image_height  = img.size
        img_array = np.asarray(Image.open(img_path))
        # print(img_array.shape)
        img_array = np.transpose(img_array, (2,0,1))
        # print(img_array.shape)
        # dataset = np.ndarray(shape=(len(img_all_data), 3, image_height, image_width),dtype=np.float32)
        h, w = image_height, image_width
        nb_pixels =  h * w
        sum_ = stat.sum
        sum_of_square = stat.sum2
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, np.sqrt(snd_moment - fst_moment ** 2)
mean, std_divide = calculate_std_mean_from_imgs()
print("mean:" + str(mean))
print("std:" + str(std_divide))


import cv2
import os

path = '../dataset/images/train2021'
means = [0, 0, 0]
stdevs = [0, 0, 0]

index = 1
num_imgs = 0
img_names = os.listdir(path)
for img_name in img_names:
    num_imgs += 1
    print(img_name)
    img = cv2.imread(os.path.join(path, img_name))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255.
    for i in range(3):
        means[i] += img[:, :, i].mean()
        stdevs[i] += img[:, :, i].std()
print(num_imgs)
means.reverse()
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

