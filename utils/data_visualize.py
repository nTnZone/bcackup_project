import os
import xml.etree.ElementTree as ET
import random
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd

testdata_path = '/home/jayce/datasets/2021/test-A-image/'
csv_path = '/home/jayce/datasets/2021/sample_submission.csv'

csv_data = pd.read_csv(csv_path)  # 读取训练数据
print(csv_data.shape)  # (189, 9)

N = 5
csv_batch_data = csv_data.head(N)  # 取后5条数据
print(csv_batch_data)
# print(csv_batch_data['name'][0])
# print(csv_batch_data['image_id'][0])
# print(csv_batch_data['confidence'][0])
# print(csv_batch_data['xmin'][0])
# print(csv_batch_data['ymin'][0])
# print(csv_batch_data['xmax'][0])
# print(csv_batch_data['ymax'][0])
print(csv_batch_data.shape)  # (5, 7)
# train_batch_data = csv_batch_data[list(range(2))]  # 取这20条数据的3到5列值(索引从0开始)
# print(train_batch_data)


print(csv_data['image_id'][39] <= 5)
# visualize first 5 images results
for i in range(csv_data.shape[0]):
    if i==0:
        print('init cv')
        image_id = str(csv_data['image_id'][i]).zfill(6)
        imgpath = testdata_path + image_id + '.jpg'
        img = cv.imread(imgpath)
        current_index = 1

    image_index = csv_data['image_id'][i]
    if image_index <= 5:
        if image_index != current_index:
            current_index = image_index
            image_id = str(csv_data['image_id'][i]).zfill(6)
            imgpath = testdata_path + image_id + '.jpg'
            img = cv.imread(imgpath)

        name = csv_data['name'][i]  # str
        confidence = str(csv_data['confidence'][i])  # float 64  ->str
        xmin = csv_data['xmin'][i]  # int64
        ymin = csv_data['ymin'][i]
        xmax = csv_data['xmax'][i]
        ymax = csv_data['ymax'][i]
        result_path = '/home/jayce/datasets/2021/testA_result/' + image_id + '.jpg'
        # print(imgpath)

        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 255), thickness=1)
        cv.putText(img, name + ' ' + confidence, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),
                   thickness=1)
        # cv.imshow('head', img)
        # save result
        cv.imwrite(result_path, img)  #

# for i in range(num):
#     pic = cv.imread(testdata_path + testimgname_list[i])