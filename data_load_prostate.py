import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
from sklearn.model_selection import StratifiedKFold
import cv2

root_dir = './Prostate_Dataset/patches_750_r7b3/'
train_dirs = ['wsi_1', 'wsi_2', 'wsi_2','wsi_3']
valid_dirs = ['wsi_4', 'wsi_5']

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


def read_prostate_dataset():
    # make whole dataset list
    # input : root that path
    # output : x_whole, y_whole that contains all file paths and classes each

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    for train_dir in train_dirs:
        train_root = root_dir + train_dir
        for(path, dir, filenames) in os.walk(train_root):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                if path[-6:] == 'benign':
                    y_class = 0
                elif path[-6:] == 'grade3':
                    y_class = 1
                elif path[-6:] == 'grade4':
                    y_class = 2
                train_x.append(file_path)
                train_y.append(y_class)


    for valid_dir in valid_dirs:
        valid_root = root_dir + valid_dir
        for(path, dir, filenames) in os.walk(valid_root):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                if path[-6:] == 'benign':
                    y_class = 0
                elif path[-6:] == 'grade3':
                    y_class = 1
                elif path[-6:] == 'grade4':
                    y_class = 2
                valid_x.append(file_path)
                valid_y.append(y_class)
        print(len(valid_x))



    print('LOADED DATA')
    print('---------# train_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          '---------# valid_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'.format(
        len(train_x), np.sum(np.asarray(train_y)==0),
        np.sum(np.asarray(train_y) == 1),
        np.sum(np.asarray(train_y) == 2),
        len(valid_x), np.sum(np.asarray(valid_y) == 0),
        np.sum(np.asarray(valid_y) == 1),
        np.sum(np.asarray(valid_y) == 2)
    )
          )


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    for i in range(0,3):
        if i == 2:
            pass
        else:
            num_dup = int(round(np.sum(train_y == 1) / np.sum(train_y == i)))
            idx = np.where(train_y == i)
            data = train_x[idx]
            labels = train_y[idx]
            for num in range(num_dup-1):
                train_x = np.concatenate([train_x, data])
                train_y = np.concatenate([train_y, labels])

    print('DUPLICATED DATA')
    print('---------# train_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'
          '---------# valid_data : {}\n'
          'benign class : {}\n'
          'cancer1 : {}\n'
          'cancer2 : {}\n'.format(
        len(train_x), np.sum(np.asarray(train_y)==0),
        np.sum(np.asarray(train_y) == 1),
        np.sum(np.asarray(train_y) == 2),
        len(valid_x), np.sum(np.asarray(valid_y) == 0),
        np.sum(np.asarray(valid_y) == 1),
        np.sum(np.asarray(valid_y) == 2)
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

# read_KBSMC_dataset()
