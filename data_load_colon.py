import os
import torch
from skimage import io, transform
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
import cv2

root_dir = './Colon_Dataset/COLON_PATCHES_1000_V2'
train_root1 = root_dir + '/train_data/v1'
train_root2 = root_dir + '/train_data/wsi'
valid_root1 = root_dir + '/valid_data/v1'
valid_root2 = root_dir + '/valid_data/wsi'
train_root = root_dir + '/train_data'
valid_root = root_dir + '/valid_data'

class ToTensor(object):

    """
    This is a transform(augmentation)class
    convert ndarrays in sample to Tensors
    """

    # swap color axis because
    # input : numpy image: H x W x C
    # output: torch image: C X H X W

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


def read_colon_dataset():
    # make whole dataset list
    # input : root that path
    # output : x_whole, y_whole that contains all file paths and classes each

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    for(path, dir, filenames) in os.walk(train_root):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            # if path[-3:] == 'wsi':
            #     y_class = 0
            # else:
            #     y_class = int(file_path[-5])
            y_class = int(file_path[-5])
            train_x.append(file_path)
            train_y.append(y_class)

    for(path, dir, filenames) in os.walk(valid_root):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            y_class = int(file_path[-5])
            valid_x.append(file_path)
            valid_y.append(y_class)



    print('LOADED DATA')
    print('---------# train_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          'cancer3 : {}\n'
          '---------# valid_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          'cancer3 : {}\n'.format(
        len(train_x), np.sum(np.asarray(train_y)==0),
        np.sum(np.asarray(train_y) == 1),
        np.sum(np.asarray(train_y) == 2)
        ,np.sum(np.asarray(train_y)==3),
        len(valid_x), np.sum(np.asarray(valid_y) == 0),
        np.sum(np.asarray(valid_y) == 1),
        np.sum(np.asarray(valid_y) == 2)
        , np.sum(np.asarray(valid_y) == 3),
    )
          )

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    for i in range(0,4):
        if i == 2:
            pass
        else:
            num_dup = int(round(np.sum(train_y == 2) / np.sum(train_y == i)))
            idx = np.where(train_y == i)
            data = train_x[idx]
            labels = train_y[idx]
            for num in range(num_dup-1):
                train_x = np.concatenate([train_x, data])
                train_y = np.concatenate([train_y, labels])

    print('DUPLECATED DATA')
    print('---------# train_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          'cancer3 : {}\n'
          '---------# valid_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          'cancer3 : {}\n'.format(
        train_x.shape[0], np.sum(train_y ==0),
        np.sum(train_y == 1),
        np.sum(train_y == 2)
        ,np.sum(train_y ==3),
        valid_x.shape[0], np.sum(valid_y == 0),
        np.sum(valid_y == 1),
        np.sum(valid_y == 2)
        , np.sum(valid_y == 3),
    )
          )

    shuffle_ix = np.arange(train_x.shape[0])
    np.random.shuffle(shuffle_ix)

    train_x = train_x[shuffle_ix]
    train_y = train_y[shuffle_ix]

    train_x = np.reshape(train_x, [train_x.shape[0], 1])
    train_y = np.reshape(train_y, [train_y.shape[0], 1])
    valid_x = np.reshape(valid_x, [valid_x.shape[0], 1])
    valid_y = np.reshape(valid_y, [valid_y.shape[0], 1])

    train_pairs = np.concatenate([train_x, train_y], axis=1).tolist()
    valid_pairs = np.concatenate([valid_x, valid_y], axis=1).tolist()

    return train_pairs, valid_pairs



