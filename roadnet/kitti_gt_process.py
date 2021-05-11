import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


train_gt_path = "../../data_road/training/gt_image_2/"
save_gt_path = "../../data_road/training/gt_image/"
gt_list = [f for f in os.listdir(train_gt_path) if f.endswith('.png')]

for gt in gt_list:
    if "road" in gt:
        img = np.array(Image.open(train_gt_path+gt))
        height = img.shape[0]
        width = img.shape[1]
        gtId = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                # print(img[i, j, :])
                if sum(img[i, j, :] == [255, 0, 255]) == 3:
                    gtId[i, j] = 7
                else:
                    gtId[i, j] = 0
        gt_name = gt.split('_road_')
        Image.fromarray(gtId).save(save_gt_path+gt_name[0]+'_'+gt_name[1])

'''
valid_gt_path = "../../data_road/validation/semantic/"
save_gt_path = "../../data_road/validation/gt_image/"
gt_list = [f for f in os.listdir(valid_gt_path) if f.endswith('.png')]

for gt in gt_list:
    img = np.array(Image.open(valid_gt_path+gt))
    height = img.shape[0]
    width = img.shape[1]
    gtId = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # print(img[i, j, :])
            if img[i, j] == 7:
                gtId[i, j] = 255
            else:
                gtId[i, j] = 0
    Image.fromarray(gtId).save(save_gt_path+gt)
'''