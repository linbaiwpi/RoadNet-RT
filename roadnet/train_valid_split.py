import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

random.seed(2020)
IMG_CROP = True

# save gt_image_2 into gt_image, so that road is assigned to 255 and non-road is 0
train_gt_path = "../../data_road/training/gt_image_2/"
save_gt_path = "../../data_road/training/gt_image/"
gt_list = [f for f in os.listdir(train_gt_path) if f.endswith('.png')]

try:
    shutil.rmtree(save_gt_path)
except OSError:
    pass
os.mkdir(save_gt_path)

pbar = tqdm(total=289)
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
        pbar.update(1)


# split the training and validation data by 9:1
def traval_split(data_path, sub='um', seed=1):
    random.seed(seed)
    data_list = [f for f in os.listdir(data_path) if sub+'_' in f]

    train_len = round(len(data_list)*0.9)
    random.shuffle(data_list)
    train_list = data_list[:train_len]
    valid_list = data_list[train_len:]
    # print(len(train_list))
    # print(len(valid_list))
    return train_list, valid_list


# load path
img_src_path = '../../data_road/training/image_2/'
gt_src_path = '../../data_road/training/gt_image/'
# save path
base_dir = '../../data_road_3/'
try:
    shutil.rmtree(base_dir)
except OSError:
    pass
os.mkdir(base_dir)
try:
    shutil.rmtree(base_dir+'training')
except OSError:
    pass
os.mkdir(base_dir+'training')
try:
    shutil.rmtree(base_dir+'validation')
except OSError:
    pass
os.mkdir(base_dir+'validation')

img_tra_path = base_dir+'training/image/'
gt_tra_path = base_dir+'training/gt_image/'
img_val_path = base_dir+'validation/image/'
gt_val_path = base_dir+'validation/gt_image/'

try:
    shutil.rmtree(img_tra_path)
except OSError:
    pass
os.mkdir(img_tra_path)
try:
    shutil.rmtree(gt_tra_path)
except OSError:
    pass
os.mkdir(gt_tra_path)
try:
    shutil.rmtree(img_val_path)
except OSError:
    pass
os.mkdir(img_val_path)
try:
    shutil.rmtree(gt_val_path)
except OSError:
    pass
os.mkdir(gt_val_path)

name_list = ['um', 'umm', 'uu']


def image_crop(img):
    return img.crop((0, int(img.size[1]*0.45), img.size[0], img.size[1]))


for name in name_list:
    train_list, valid_list = traval_split(img_src_path, sub=name)
    for valid_img in valid_list:
        if IMG_CROP:
            img = Image.open(img_src_path+valid_img)
            img_crop = image_crop(img)
            img_crop.save(img_val_path+valid_img)

            gt = Image.open(gt_src_path+valid_img)
            gt_crop = image_crop(gt)
            gt_crop.save(gt_val_path+valid_img)
        else:
            shutil.copy(img_src_path+valid_img, img_val_path+valid_img)
            shutil.copy(gt_src_path+valid_img, gt_val_path+valid_img)
    for train_img in train_list:
        if IMG_CROP:
            img = Image.open(img_src_path+train_img)
            img_crop = image_crop(img)
            img_crop.save(img_tra_path+train_img)

            gt = Image.open(gt_src_path+train_img)
            gt_crop = image_crop(gt)
            gt_crop.save(gt_tra_path+train_img)
        else:
            shutil.copy(img_src_path+train_img, img_tra_path+train_img)
            shutil.copy(gt_src_path+train_img, gt_tra_path+train_img)

