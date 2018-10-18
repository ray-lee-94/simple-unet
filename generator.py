import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import glob
import os
import keras

ia.seed(1)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=32, dim=(576, 576), n_channel=1, n_class=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.n_class = n_class
        self.shuffle = shuffle
        self.image_path = './images/train/images/'
        self.label_path = './images/train/label/'
        self.lis_names = [x.split('/')[-1].split('.')[0] for x in glob.glob(self.image_path + '*.*')]
        self.aug=iaa.Sequential(
            [iaa.ElasticTransformation(alpha=50,sigma=5),iaa.Fliplr(p=0.5),iaa.Flipud(p=0.5)],random_order=True)
        # self.aug = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.lis_names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_names = [self.lis_names[i] for i in indexes]
        x, y = self.__data_generation(batch_names)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.lis_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_names):
        x = np.empty((self.batch_size, *self.dim, self.n_channel))
        y = np.empty((self.batch_size, *self.dim, self.n_channel))

        for i, id in enumerate(list_names):
            padded_image = self.__load_image(id, target_shape=576)
            padded_label = self.__load_label(id, target_shape=576)
            if self.aug:
                aug_det = self.aug.to_deterministic()
                padded_image = aug_det.augment_image(padded_image)
                padded_label = aug_det.augment_image(padded_label)
                padded_label[padded_label>0]=1

            x[i, ...] = np.expand_dims(padded_image, axis=-1)
            y[i, ...] = np.expand_dims(padded_label, axis=-1)
            # cv2.imshow(" ",np.array(y[i, ...]))
            # cv2.waitKey(10000)
            # cv2.imshow("",np.array( padded_label,dtype=np.uint8))
            # cv2.waitKey(10000)
        return x, y

    def __load_image(self, image, target_shape=576):
        img = cv2.imread(os.path.join(self.image_path, image + '.tif'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        padded_img = padding(img, target_shape=target_shape)
        return np.array(padded_img, dtype=np.float32)

    def __load_label(self, image, target_shape=576):
        img = cv2.imread(os.path.join(self.label_path, image + '.tif'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        padded_img = padding(img, target_shape=target_shape)
        padded_img = np.array(padded_img, dtype=np.float32)
        padded_img[padded_img > 0] = 1
        return padded_img


def padding(image, target_shape):
    org_shape = image.shape[0]
    border = int((target_shape - org_shape) / 2)
    padded_img = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT)
    return padded_img


def cropping(image,target_shape):
    org_shape=image.shape[0]
    border=int((org_shape-target_shape)/2)
    cropped_image=image[border:target_shape-border,border:target_shape-border,]
    return cropped_image
