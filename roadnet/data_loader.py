import numpy as np
import os
import cv2
import keras

import matplotlib.pyplot as plt

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    #CLASSES = ['sky',       'building',     'pole',         'road',
    #           'pavement',  'tree',         'signsymbol',   'fence',
    #           'car',       'pedestrian',   'bicyclist',    'unlabelled']

    CLASSES = ['unlabeled', 'ego vehicle', 'rectification border',
               'out of roi', 'static', 'dynamic',
               'ground', 'road']

    def __init__(
            self,
            images_dir,
            masks_dir,
            shape=(None, None),
            classes=None,
            augmentation=None,
            preprocessing=None,
            add_location=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        # print(classes)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.add_location = add_location

        self.shape = shape

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.shape, interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)

        #plt.imshow(image)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        #print(image.shape)
        #print(mask.shape)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.add_location:
            '''
            ch_x = np.zeros((image.shape[0], image.shape[1], 1))
            ch_y = np.zeros((image.shape[0], image.shape[1], 1))

            for i in range(image.shape[0]):
                ch_x[i, :] = np.expand_dims(np.array(range(image.shape[1]))/image.shape[1], -1)

            for j in range(image.shape[1]):
                ch_y[:, j] = np.expand_dims(np.array(range(image.shape[0]))/image.shape[0], -1)

            np.save("ch_xy", np.concatenate((ch_x, ch_y), axis=2))
            '''
            ch_xy = np.load(self.add_location)
            image = np.concatenate((image, ch_xy), axis=2)

        return image, mask

        #print(image.shape)
        #print(mask.shape)

    def __len__(self):
        return len(self.ids)


class Dataset_CamVid:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    CLASSES = ['sky',       'building',     'pole',         'road',
               'pavement',  'tree',         'signsymbol',   'fence',
               'car',       'pedestrian',   'bicyclist',    'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            shape=(None, None),
            classes=None,
            augmentation=None,
            preprocessing=None,
            add_location=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        # print(classes)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.add_location = add_location

        self.shape = shape

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.shape, interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)

        #plt.imshow(image)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        #print(image.shape)
        #print(mask.shape)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.add_location:
            '''
            ch_x = np.zeros((image.shape[0], image.shape[1], 1))
            ch_y = np.zeros((image.shape[0], image.shape[1], 1))

            for i in range(image.shape[0]):
                ch_x[i, :] = np.expand_dims(np.array(range(image.shape[1]))/image.shape[1], -1)

            for j in range(image.shape[1]):
                ch_y[:, j] = np.expand_dims(np.array(range(image.shape[0]))/image.shape[0], -1)

            np.save("ch_xy", np.concatenate((ch_x, ch_y), axis=2))
            '''
            ch_xy = np.load(self.add_location)
            image = np.concatenate((image, ch_xy), axis=2)

        return image, mask

        #print(image.shape)
        #print(mask.shape)

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        # print("batch length: " + str(len(batch)))
        # print("batch shape: " + str(batch[0].shape))
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
