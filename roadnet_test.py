import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from shutil import rmtree

#import cv2
import json
import keras
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

import segmentation_models as sm

from roadnet.utils import visualize, denormalize
from roadnet.data_loader import Dataset, Dataloder
from roadnet.data_aug import get_training_augmentation, get_preprocessing, get_validation_augmentation
from roadnet.net import roadnet_rt

DATA_DIR = '../data_road_2/'
# DATA_DIR = '../camvid/data/CamVid/'
x_test_dir = os.path.join(DATA_DIR, 'testing/image_2/')
y_test_dir = os.path.join(DATA_DIR, 'testing/image_2/')


config_path = "./roadnet/config.json"
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

BACKBONE = config["model"]["BACKBONE"]
CLASSES = config["model"]["CLASSES"].split("delimiter")
BATCH_SIZE = config["train"]["BATCH_SIZE"]
LR = config["train"]["LR"]
EPOCHS = config["train"]["EPOCHS"]
HEIGHT = 280 #352
WIDTH = 960 #1216
# HEIGHT = config["model"]["IN_HEIGHT"]
# WIDTH = config["model"]["IN_WIDTH"]
path_test_weight = './roadnet_rt_9257.hdf5'
path_cmp_weight = './roadnet_rt_9257.hdf5'

# preprocess_input = sm.get_preprocessing(BACKBONE)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    shape=(WIDTH, HEIGHT),
    classes=CLASSES,
    preprocessing=get_preprocessing() # , add_location='ch_xy_352.npy'
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
add_location = False
add_crop = False
model = roadnet_rt((HEIGHT, WIDTH), n_classes, activation).build()
model.summary()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# load test weights
model.load_weights('./roadnet_rt_9257.hdf5')

optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

model.compile(optim, total_loss, metrics)

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()

    visualize(
        image=denormalize(image.squeeze()),
        pr_mask=pr_mask[..., 0].squeeze(),
    )


def image_crop(img):
    img_crop = img.crop((0, int(img.size[1]*0.45), img.size[0], img.size[1]))
    img_shape = img_crop.size
    return img_crop, img_shape


def image_pad(img, img_shape):
    img_pad = np.zeros((img_shape[1], img_shape[0]))
    img_pad[(img_shape[1]-img.size[1]):img_shape[1], 0:img_shape[0]] = np.array(img)
    return img_pad

# visualize all the prediction and save prediction
pred_msk_dir = "./pred_mask/"
pred_vis_dir = "./pred_visual/"

try:
    rmtree(pred_msk_dir)
except OSError:
    pass
try:
    rmtree(pred_vis_dir)
except OSError:
    pass
os.mkdir(pred_msk_dir)
os.mkdir(pred_vis_dir)


test_img_list = os.listdir(x_test_dir)
for img_name in test_img_list:
    img = Image.open(x_test_dir+img_name)
    img_shape = img.size
    if add_crop:
        img_e, crop_shape = image_crop(img)
    else:
        img_e = img
    img_e = np.array(img_e.resize((WIDTH, HEIGHT), Image.BILINEAR))
    img = np.array(img)
    img_e = img_e / 255
    if add_location:
        ch_xy = np.load('ch_xy_352.npy')
        img_e = np.concatenate((img_e, ch_xy), axis=2)
    img_e = np.expand_dims(img_e, 0)
    pred_msk = np.squeeze(model.predict(img_e, verbose=1)*255).round()
    if add_crop:
        pred_msk = Image.fromarray(pred_msk).resize((crop_shape[0], crop_shape[1]), Image.NEAREST)
        pred_msk_pad = image_pad(pred_msk, img_shape)
        # print(pred_msk_pad.shape)
    else:
        pred_msk_pad = np.array(Image.fromarray(pred_msk).resize(img_shape, Image.NEAREST))
    # save mask
    pred_msk_pad = pred_msk_pad.astype(np.uint8)
    Image.fromarray(pred_msk_pad).save(pred_msk_dir+img_name.split('_')[0]+'_road_'+img_name.split('_')[1])
    # visualize
    img[:, :, 0] = np.bitwise_or(img[:, :, 0], pred_msk_pad)
    Image.fromarray(img).save(pred_vis_dir+img_name)

# visualize training and validation set
x_train_dir = os.path.join(DATA_DIR, 'training/image_2/')
x_valid_dir = os.path.join(DATA_DIR, 'validation/image_2/')
train_vis_dir = "./train_visual/"
valid_vis_dir = "./valid_visual/"

try:
    rmtree(train_vis_dir)
except OSError:
    pass
try:
    rmtree(valid_vis_dir)
except OSError:
    pass
os.mkdir(train_vis_dir)
os.mkdir(valid_vis_dir)

train_img_list = os.listdir(x_train_dir)
for img_name in train_img_list:
    img = Image.open(x_train_dir+img_name)
    img_shape = img.size
    if add_crop:
        img_e, crop_shape = image_crop(img)
    else:
        img_e = img
    img_e = np.array(img_e.resize((WIDTH, HEIGHT), Image.BILINEAR))
    img = np.array(img)
    img_e = img_e / 255
    if add_location:
        ch_xy = np.load('ch_xy_352.npy')
        img_e = np.concatenate((img_e, ch_xy), axis=2)
    img_e = np.expand_dims(img_e, 0)
    pred_msk = np.squeeze(model.predict(img_e, verbose=1).round()*255)
    if add_crop:
        pred_msk = Image.fromarray(pred_msk).resize((crop_shape[0], crop_shape[1]), Image.NEAREST)
        pred_msk_pad = image_pad(pred_msk, img_shape)
        # print(pred_msk_pad.shape)
    else:
        pred_msk_pad = np.array(Image.fromarray(pred_msk).resize(img_shape, Image.NEAREST))
    pred_msk_pad = pred_msk_pad.astype(np.uint8)
    # visualize
    img[:, :, 0] = np.bitwise_or(img[:, :, 0], pred_msk_pad)
    Image.fromarray(img).save(train_vis_dir+img_name.split('_')[0]+'_road_'+img_name.split('_')[1])


valid_img_list = os.listdir(x_valid_dir)
for img_name in valid_img_list:
    img = Image.open(x_valid_dir+img_name)
    img_shape = img.size
    if add_crop:
        img_e, crop_shape = image_crop(img)
    else:
        img_e = img
    img_e = np.array(img_e.resize((WIDTH, HEIGHT), Image.BILINEAR))
    img = np.array(img)
    img_e = img_e / 255
    if add_location:
        ch_xy = np.load('ch_xy_352.npy')
        img_e = np.concatenate((img_e/255, ch_xy), axis=2)
    img_e = np.expand_dims(img_e, 0)
    pred_msk = np.squeeze(model.predict(img_e, verbose=1).round()*255)
    if add_crop:
        pred_msk = Image.fromarray(pred_msk).resize((crop_shape[0], crop_shape[1]), Image.NEAREST)
        pred_msk_pad = image_pad(pred_msk, img_shape)
        # print(pred_msk_pad.shape)
    else:
        pred_msk_pad = np.array(Image.fromarray(pred_msk).resize(img_shape, Image.NEAREST))
    pred_msk_pad = pred_msk_pad.astype(np.uint8)
    # visualize
    img[:, :, 0] = np.bitwise_or(img[:, :, 0], pred_msk_pad)
    Image.fromarray(img).save(valid_vis_dir+img_name.split('_')[0]+'_road_'+img_name.split('_')[1])


# compare the mask
cmp_msk_dir = "./cmp_mask/"
try:
    rmtree(cmp_msk_dir)
except OSError:
    pass
os.mkdir(cmp_msk_dir)

# HEIGHT = 160
# WIDTH = 600

model2cmp = BiSeNet_mod4_base3((HEIGHT, WIDTH), n_classes, activation).build()
model2cmp.compile(optim, total_loss, metrics)
model2cmp.load_weights(path_cmp_weight)

test_img_list = os.listdir(x_test_dir)
for img_name in test_img_list:
    img = Image.open(x_test_dir+img_name)
    img_shape = img.size
    if add_crop:
        img_e, crop_shape = image_crop(img)
    else:
        img_e = img
    img_e = np.array(img_e.resize((WIDTH, HEIGHT), Image.BILINEAR))
    img = np.array(img)
    img_e = img_e / 255
    if add_location:
        ch_xy = np.load('ch_xy_352.npy')
        img_e = np.concatenate((img_e/255, ch_xy), axis=2)
    img_e = np.expand_dims(img_e, 0)
    pred_msk = np.squeeze(model2cmp.predict(img_e, verbose=1).round()*255)
    pred_msk = pred_msk.astype(np.uint8)
    if add_crop:
        pred_msk = Image.fromarray(pred_msk).resize((crop_shape[0], crop_shape[1]), Image.NEAREST)
        pred_msk_pad = image_pad(pred_msk, img_shape)
    else:
        pred_msk_pad = np.array(Image.fromarray(pred_msk).resize(img_shape, Image.NEAREST))
    pred_msk_pad = pred_msk_pad.astype(np.uint8)
#    if True:
#        plt.imshow(pred_msk)
#        plt.show()
    # visualize
    img = np.array(Image.open(pred_vis_dir+img_name))
#    if True:
#        plt.imshow(img)
#        plt.show()
    img[:, :, 1] = np.bitwise_or(img[:, :, 1], pred_msk_pad)
#    if True:
#        plt.imshow(img)
#        plt.show()
    Image.fromarray(img).save(cmp_msk_dir+img_name)

    # yellow: overlap
    # green: baseline
    # red: add-on
