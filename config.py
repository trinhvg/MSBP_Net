import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import numpy as np

import importlib
import torch.optim as optim



####
class Config(object):
    def __init__(self, _args=None):
        if _args is not None:
            self.__dict__.update(_args.__dict__)

        self.nr_epochs = 60
        self.train_batch_size = 4
        self.infer_batch_size = 24
        self.seed = 5
        self.optimizer =[
            optim.Adam,
            {  # should match keyword for parameters within the optimizer
                'lr': 1.0e-4,  # initial learning rate,
                # 'weight_decay' : 0.02
            }
        ],
        self.scheduler = lambda x: optim.lr_scheduler.StepLR(x, 60),  # learning rate scheduler
        self.dataset = self.dataset
        self.network_name = self.network_name
        self.saved_path = self.saved_path
        if self.dataset == 'colon_tma':
            self.nr_class = 4
            self.data_size = [1024, 1024]
            self.input_size = [1024, 1024]
        else:
            self.nr_class = 3
            self.data_size  = [750, 750]
            self.input_size = [512, 512]

        # nr of processes for parallel processing input
        self.nr_procs_train = 4
        self.nr_procs_valid = 4


        self.logging = False  # False for debug run to test code
        self.log_path = f'/data1/trinh/data/predicted_data/MSBP/{self.dataset}'

        self.chkpts_prefix = 'model'
        # self.model_name = f'{self.exp_mode}'
        self.log_dir = self.log_path + self.network_name + f'_seed{self.seed}/'
        print(self.log_dir)

    def train_augmentors(self):
        shape_augs = [
            iaa.PadToFixedSize(
                self.data_size[0],
                self.data_size[1],
                pad_cval=255,
                position='center').to_deterministic(),
            iaa.Affine(
                cval=255,
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2),
                       "y": (0.8, 1.2)},
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.01, 0.01),
                                   "y": (-0.01, 0.01)},
                rotate=(-179, 179),  # rotate by -179 to +179 degrees
                shear=(-5, 5),  # shear by -5 to +5 degrees
                order=[0],  # use nearest neighbour
                backend='cv2'  # opencv for fast processing
            ),
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            iaa.CropToFixedSize(self.input_size[0],
                                self.input_size[1],
                                position='center').to_deterministic()
        ]
        #
        input_augs = [
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
                iaa.MedianBlur(k=(3, 5)),  # median with random kernel sizes
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            ]),
            iaa.Sequential([
                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
            ], random_order=True),
        ]
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.PadToFixedSize(
                self.data_size[0],
                self.data_size[1],
                pad_cval=255,
                position='center').to_deterministic(),
            iaa.CropToFixedSize(self.input_size[0],
                                self.input_size[1],
                                position='center').to_deterministic()
        ]
        return shape_augs, None

############################################################################
