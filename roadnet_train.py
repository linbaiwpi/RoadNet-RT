import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import json
import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K

import segmentation_models as sm

from roadnet.utils import visualize
from roadnet.data_loader import Dataset, Dataloder
from roadnet.data_aug import get_training_augmentation, get_preprocessing, get_validation_augmentation
from roadnet.clr_callback import CyclicLR
from roadnet.learningratefinder import LearningRateFinder
from roadnet.net import roadnet_rt

# json reading
config_path = "./roadnet/config.json"
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

DATA_DIR = '../data_road_2/'
# DATA_DIR = '../camvid/data/CamVid/'

'''
# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')
'''

x_train_dir = os.path.join(DATA_DIR, 'training/image_2')
y_train_dir = os.path.join(DATA_DIR, 'training/gt_image')

x_valid_dir = os.path.join(DATA_DIR, 'validation/image_2')
y_valid_dir = os.path.join(DATA_DIR, 'validation/gt_image')

# reading the json configuration file
BACKBONE = config["model"]["BACKBONE"] # used for naming h5 only
CLASSES = config["model"]["CLASSES"].split("delimiter")[0].split(',')
BATCH_SIZE = config["train"]["BATCH_SIZE"]
LR = config["train"]["LR"]
EPOCHS = config["train"]["EPOCHS"]
HEIGHT = config["model"]["IN_HEIGHT"]
WIDTH = config["model"]["IN_WIDTH"]

# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, shape=(WIDTH, HEIGHT), classes=['road'])
image, mask = dataset[5]  # get some sample
# visualize(
#     image=image,
#     cars_mask=mask[..., 0].squeeze()
# )

# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, shape=(WIDTH, HEIGHT), classes=['road'],
                  # augmentation=get_training_augmentation(height=HEIGHT, width=WIDTH),
                  preprocessing=get_preprocessing())
image, mask = dataset[12]  # get some samplesample
# print("image.shape AFTER: "+str(image.shape))
# plt.imshow(image)
# plt.show()
# visualize(
#     image=image,
#     cars_mask=mask[..., 0].squeeze()
# )

'''
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['road']
LR = 0.0001
EPOCHS = 1
'''
#CLASSES = ['car']

# preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = roadnet_rt((HEIGHT, WIDTH), n_classes, activation).build()
model.summary()

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss


def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

iou = sm.metrics.IOUScore(threshold=0.5)
f1 = sm.metrics.FScore(threshold=0.5)
metrics = [iou, f1]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    shape=(WIDTH, HEIGHT),
    classes=CLASSES,
    augmentation=get_training_augmentation(height=HEIGHT, width=WIDTH),
    preprocessing=get_preprocessing()
)
'''
image, mask = train_dataset[0]
print("trainig set shape: "+str(image.shape))
print("trainig set shape: "+str(mask.shape))
print("image value ranges from "+str(image.min())+" to "+str(image.max()))
plt.imshow(image)
plt.show()
'''

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    shape=(WIDTH, HEIGHT),
    classes=CLASSES,
    augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
    preprocessing=get_preprocessing()
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
# image, mask = train_dataset[0]
# print("trainig set shape: "+str(image.shape))
# print("trainig set shape: "+str(mask.shape))
assert train_dataloader[0][0].shape == (BATCH_SIZE, HEIGHT, WIDTH, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, HEIGHT, WIDTH, n_classes)
# image, mask = valid_dataset[0]
# print("valid set shape: "+str(image.shape))
# print("valid set shape: "+str(mask.shape))
assert valid_dataloader[0][0].shape == (1, HEIGHT, WIDTH, 3)
assert valid_dataloader[0][1].shape == (1, HEIGHT, WIDTH, n_classes)

STEP_SIZE = 8
CLR_METHOD = "triangular"
MIN_LR = 5e-4
MAX_LR = 5e-3
stepSize = STEP_SIZE * (289 // BATCH_SIZE)
clr = CyclicLR(mode=CLR_METHOD,	base_lr=MIN_LR,	max_lr=MAX_LR, step_size=stepSize)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./h5/'+BACKBONE+'weights.{epoch:03d}-{val_loss:.3f}.hdf5',
                                    save_weights_only=True, save_best_only=True,
                                    monitor='val_f1-score', mode='max', verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30, verbose=1, mode='min', min_lr=1e-10),
    keras.callbacks.EarlyStopping(monitor='val_f1-score', patience=200, verbose=1, mode='max'),
]
'''
print("[INFO] finding learning rate...")
lrf = LearningRateFinder(model)
lrf.find(
    train_dataloader,
    1e-10, 1e+1,
    stepsPerEpoch=np.ceil((289 / float(BATCH_SIZE))),
    batchSize=BATCH_SIZE)
# plot the loss for the various learning rates and save the
# resulting plot to disk
lrf.plot_loss()
'''

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)
# for key, val in history.history.items():
#     print(key)
#     print(val)
# Plot training & validation iou_score values
# plt.figure(figsize=(30, 5))
# plt.subplot(121)

plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score'+', max='+str(max(history.history['val_iou_score'])))
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('iou_'+BACKBONE+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(LR)+'_'+str(HEIGHT)+'_'+str(WIDTH)+'.png')
plt.show()

# Plot training & validation loss values
# plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss'+', min='+str(min(history.history['val_loss'])))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('loss_'+BACKBONE+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(LR)+'_'+str(HEIGHT)+'_'+str(WIDTH)+'.png')
plt.show()

plt.plot(history.history['lr'])
plt.title('Change of learning rate')
plt.ylabel('lr')
plt.xlabel('Epoch')
plt.savefig('lr_'+BACKBONE+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(LR)+'_'+str(HEIGHT)+'_'+str(WIDTH)+'.png')
plt.show()
