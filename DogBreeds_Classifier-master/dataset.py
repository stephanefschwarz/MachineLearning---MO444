import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from data_augmentation import random_transform, transform
import keras.utils


class Dataset(keras.utils.Sequence):
    '''Class that gets a folder path and from that generates
    a dataframe containing each image path and
    label (retrieved from img name). Inherits from Sequence,
    which enables to be use with keras fir_generator method.
    Loads image dynamically to memory, perform augmentations
    and allows CPU multiprocessing while training with GPU.
    Input param 'augmentation' is either None or
    a dict containing the ranges for each transform:
            (rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            horizontal_flip=False,
            vertical_flip=False)'''

    def __init__(self, path, shuffle=False,
                 batch_size=32, tgt_size=(299, 299),
                 color=True, augmentation=None, len_mod=1, load_labels_from_file=False):
        if type(path) is list:
            self.path = path
        else:
            self.path = [path]
        self.batch_size = batch_size
        self.tgt_size = tgt_size
        self.color = color
        self.augmentation = augmentation
        self.len_mod = len_mod
        self.load_labels_from_file = load_labels_from_file

        self.dir2df()
        self.x = self.df['path'].values
        self.y = self.onehot_y()
        if shuffle:
            perm_array = np.arange(len(self.x))
            np.random.shuffle(perm_array)
            self.x = self.x[perm_array]
            self.y = self.y[perm_array]

    def __len__(self):
        return int(self.len_mod * np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.normalize(self.load_images(batch_x))
        if self.augmentation is not None:
            batch_x = self.randomly_augment_images(batch_x)
        return batch_x, batch_y

    def load_images(self, paths):
        return np.array([self.read(p) for p in paths])

    def normalize(self, images):
        return np.array([img / img.max() for img in images])

    def randomly_augment_images(self, images):
        return np.array([random_transform(im, **self.augmentation) for
                         im in images])

    def augment_images(self, images, augmentations):
        return np.array([transform(im, **augmentations) for
                         im in images])

    def read(self, path):
        im = cv2.imread(
            path, cv2.IMREAD_COLOR if self.color else cv2.IMREAD_GRAYSCALE)
        if self.tgt_size is not None:
            im = cv2.resize(im, self.tgt_size)
        return im

    def onehot_y(self):
        self.encoder = OneHotEncoder()
        y = self.df['label'].values.astype(np.float64).reshape(-1, 1)
        return self.encoder.fit_transform(y).toarray()

    def dir2df(self):
        if not self.load_labels_from_file:
            dfs = []
            for path in self.path:
                walk = list(os.walk(path))[0]
                fname = [x for x in walk[-1] if x.lower().endswith('.jpg')]
                paths = [os.path.join(walk[0], x) for x in fname]
                tmp = [name.split('/')[-1] for name in fname]
                labels = [x.split('_')[0] for x in tmp]
                ids = [x.split('_')[-1].split('.')[0] for x in tmp]
                dfs.append(pd.DataFrame(
                    dict(path=paths, label=labels, id=ids)))
            self.df = pd.concat(dfs, axis=0)

        else:
            with open(self.load_labels_from_file, 'r') as f:
                tmp = [l.strip('\n').split('/')[-1].split(' ') for l in f]
            ids = [x[0] for x in tmp]
            labels = [x[1] for x in tmp]
            paths = [os.path.join(self.path[0], idx) for idx in ids]
            self.df = pd.DataFrame(dict(path=paths, label=labels, id=ids))

        self.df.set_index('id', inplace=True)

    def analyze_df(self):
        tmp = []
        for path in tqdm(self.df['path'].values):
            tmp.append(self.read(path).shape)
        self.df['height'] = [x[0] for x in tmp]
        self.df['width'] = [x[1] for x in tmp]
        print(self.df.describe())

    def TTA(self, models, batch_size, augmentations, len_reducersp=1):
        '''models is a list of keras model
        augmentation is a list of augmentation dicts following keras struct'''
        modelwise_tmp = []
        for model in models:
            augwise_tmp = []
            for aug in augmentations:
                batchwise_tmp = []
                for i in tqdm(range(int(self.__len__() * len_reducer))):
                    batch_x = self.x[i * batch_size:(i + 1) * batch_size]
                    batch_x = self.normalize(self.load_images(batch_x))
                    if aug is not None:
                        batch_x = self.augment_images(batch_x, aug)

                    batchwise_tmp.append(model.model.predict(x=batch_x))
                augwise_tmp.append(np.concatenate(batchwise_tmp, axis=0))
            modelwise_tmp.append(np.moveaxis(np.array(augwise_tmp), 0, -1))
        pred = np.moveaxis(np.array(modelwise_tmp), 0, -1)

        batchwise_tmp = []
        for i in range(int(self.__len__() * len_reducer)):
            batch_y = self.y[i * batch_size:(i + 1) * batch_size]
            batchwise_tmp.append(batch_y)
        y = np.concatenate(batchwise_tmp, axis=0)

        return pred, y


if __name__ == '__main__':
    TRAIN_PATH = '/media/data/MOA144/Assignment 4/train'
    train = Dataset(TRAIN_PATH)
    train.analyze_df()
