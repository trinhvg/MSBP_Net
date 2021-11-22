import os
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import importlib

import pandas as pd
import numpy as np
import openslide
from imgaug import augmenters as iaa
from progress.bar import Bar as ProgressBar  # Easy progress reporting for Python

from infer_wsi_utils import *
from config import Config
from define_network import define_network


class Inferer(Config):
    def __init__(self, _args=None):
        super(Inferer, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
        self.project_path = '/data1/trinh/data/raw_data/KBSMC/Colon/Colon_WSI/'
        self.in_img_path = f'{self.project_path}/image/ColonWSI/'
        self.in_ano_path = f'{self.project_path}/label/Colon_WSI_annotation_npy_v0/'
        self.out_img_path = f'/data1/trinh/data/predicted_data/SBP_pred_npy_smaller_stride_prenet/ResNet_v2/'
        self.infer_batch_size = 64
        self.nr_procs_valid = 31
        self.patch_size = 1024
        self.patch_stride = 1024
        self.nr_classes = 4

    def resize_save(self, svs_code, save_name, img, scale=1.0):
        ano = img.copy()
        cmap = plt.get_cmap('jet')
        path = f'{self.out_img_path}/{svs_code}/'
        img = (cmap(img / scale)[..., :3] * 255).astype('uint8')
        img[ano == 0] = [10, 10, 10]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{path}/{save_name}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return 0

    def infer_step_c(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_class = net(imgs)  # forward
            if 'msd' in net_name:
                logit_class = logit_class[-1]
            prob = nn.functional.softmax(logit_class, dim=1)
            # prob = prob.permute(0, 2, 3, 1)  # to NHWC
            return prob.cpu().numpy()

    def predict_one_model(self, net, svs_code, net_name='Mob_add'):
        try:
            slide = openslide.OpenSlide(f'{self.in_img_path}/{svs_code}.ndpi')
        except:
            slide = openslide.OpenSlide(f'{self.in_img_path}/{svs_code}.svs')

        ano_list = read_ano_text(f'{self.in_ano_path.replace("npy", "txt")}/{svs_code}_ano.txt')
        roi = find_roi(ano_list)
        ano = np.float32(np.load(f'{self.in_ano_path}/{svs_code}.npy'))  # [h, w]
        patch_list = generate_patch_list(ano, roi, self.patch_size, self.patch_stride)

        inf_output_dir = f'{self.out_img_path}/{svs_code}/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir)

        infer_augmentors = self.infer_augmentors()
        infer_dataset = DatasetSerialPatch(slide, patch_list, self.patch_size,
                                           shape_augs=iaa.Sequential(infer_augmentors[0]),
                                           input_augs=iaa.Sequential(infer_augmentors[1]))

        dataloader = data.DataLoader(infer_dataset,
                                     num_workers=self.nr_procs_valid,
                                     batch_size=self.infer_batch_size,
                                     shuffle=False,
                                     drop_last=False)

        out_prob = np.zeros([self.nr_classes, ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]
        out_prob_count = np.zeros([ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]

        for batch_data in dataloader:
            imgs_input, imgs_path = batch_data
            output_prob = self.infer_step_c(net, imgs_input, net_name)
            for idx, patch_loc in enumerate(imgs_path):
                patch_loc = np.array(eval(patch_loc)) // 16
                for grade in range(self.nr_classes):
                    out_prob[grade][patch_loc[0]:patch_loc[0] + self.patch_size // 16,
                    patch_loc[1]:patch_loc[1] + self.patch_size // 16] += output_prob[idx][grade]
                    out_prob_count[patch_loc[0]:patch_loc[0] + self.patch_size // 16,
                    patch_loc[1]:patch_loc[1] + self.patch_size // 16] += 1

        out_prob_count[out_prob_count == 0.] = 4.
        out_prob_count /= 4.
        out_prob /= out_prob_count
        predict = np.argmax(out_prob, axis=0) + 1
        # plt.imshow(predict)
        # plt.show()
        predict_2 = predict.copy()

        for c in range(self.nr_classes):
            out_prob[c][ano == 0] = 0
        predict[ano == 0] = 0

        predict_2[ano == 0] = 5

        unique, counts = np.unique((ano - predict_2), return_counts=True)
        ano_count = dict(zip(unique, counts))
        acc = int(ano_count[0.0] / (ano.shape[0] * ano.shape[1] - ano_count[-5]) * 10000)
        f1 = compute_f1(predict, np.uint(ano))

        self.resize_save(svs_code, f'predict_{net_name}_{acc}_{f1}', predict, scale=4.0)
        print(f'predict_{net_name}_{acc}_{f1}')
        self.resize_save(svs_code, 'ano', ano, scale=4.0)
        np.save(f'{self.out_img_path}/{svs_code}/predict_{net_name}', predict)
        np.save(f'{self.out_img_path}/{svs_code}/ano', ano)
        print('done')
        return 0

    def run_wsi(self, ):

        net_name = self.network_name
        print(net_name)

        net = define_network(self.network_name, self.nr_class)
        net = torch.nn.DataParallel(net).to('cuda')

        saved_state = torch.load(self.saved_path)
        net.load_state_dict(saved_state, strict=True)

        # #-------------------------------------------------------------------------------------------------------------
        name_wsi_list = findExtension(self.in_ano_path, '.npy')

        for name in name_wsi_list:
            svs_code = name
            print(svs_code)
            acc_wsi = []
            acc_one_model = self.predict_one_model(net, svs_code, net_name=net_name)
            acc_wsi.append(acc_one_model)


####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--dataset', type=str, default='colon_tma', help='colon_tma, prostate_tma')
    parser.add_argument('--network_name', type=str, default='VGG', help='ResNet, MobileNetV1, EfficientNet, VGG, ResNeSt'
                                                                        'MuDeep, MSDNet, Res2Net'
                                                                        'ResNet_MSBP, ResNet_add, ResNet_conv, ResNet_concat'
                                                                        'ResNet_concat_zm, ResNet_conv_zm')
    parser.add_argument('--saved_path', type=str, default='', help='path to trained models to validate')

    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    inferer = Inferer(_args=args)
    inferer.run_wsi()
