# 继承pytorch的dataset，创建自己的

import torch
from torch import nn
from torch.nn import functional as F
import ttach as tta
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
import pandas as pd
import numpy as np
import torchvision
from sklearn.model_selection import KFold
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook as tqdm
import timm



class TrainValidData(torch.utils.data.Dataset):
    def __init__(self, csv_path, imag_path, transform=None):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
        """
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.imag_path = imag_path
        self.transform = transform
        self.class_to_num={'airplane': 0,
 'automobile': 1,
 'bird': 2,
 'cat': 3,
 'deer': 4,
 'dog': 5,
 'frog': 6,
 'horse': 7,
 'ship': 8,
 'truck': 9}
        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path)
        # 文件第一列包含图像文件名称
        self.image_arr = np.asarray(self.data_info.iloc[0:, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始一直读取到最后一行
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[0:, 1])
        # 计算 length
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = str(int(self.image_arr[index])) + '.png'

        # 读取图像文件
        img_as_img = Image.open(self.imag_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        # if img_as_img.mode != 'L':
        #     img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        transform = self.transform
        img_as_img = transform(img_as_img)

        # 得到图像的 label
        label = self.label_arr[index]
        number_label = self.class_to_num[label]

        return (img_as_img, number_label)  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len