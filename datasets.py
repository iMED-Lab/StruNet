# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF


class CT_loader(data.Dataset):
    def __init__(self, data_dir, channel=1, isTraining=True):
        super(CT_loader, self).__init__()
        self.low_lst, self.high_lst = self.get_dataPath(data_dir, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.name = ""
    
    def __getitem__(self, index):
        low_path = self.low_lst[index]
        self.name = low_path.split("/")[-1]
        high_path = self.high_lst[index]
        simple_transform = transforms.ToTensor()
        
        low = Image.open(low_path)
        high = Image.open(high_path)
        
        if self.channel == 1:
            low = low.convert("L")
            high = high.convert("L")
            norm_transform = transforms.Normalize((0.5,), (0.5,))
        else:
            low = low.convert("RGB")
            high = high.convert("RGB")
            norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            low = low.rotate(angel)
            high = high.rotate(angel)
        
        low = norm_transform(simple_transform(low))
        high = norm_transform(simple_transform(high))
        
        return low, high
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.low_lst)
    
    def get_dataPath(self, data_dir, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param data_dir: 存放的文件夹
        :return:
        """
        if isTraining:
            low_dir = os.path.join(data_dir, "train", "1E4")
            high_dir = os.path.join(data_dir, "train", "high")
        else:
            low_dir = os.path.join(data_dir, "test", "1E4")
            high_dir = os.path.join(data_dir, "test", "high")
        
        low_lst = sorted(list(map(lambda x: os.path.join(low_dir, x), os.listdir(low_dir))))
        high_lst = sorted(list(map(lambda x: os.path.join(high_dir, x), os.listdir(high_dir))))
        
        assert len(low_lst) == len(high_lst)
        
        return low_lst, high_lst
    
    def getFileName(self):
        return self.name


class OCT_loader(data.Dataset):
    def __init__(self, data_dir, channel=1, scale_size=512, isTraining=True):
        super(OCT_loader, self).__init__()
        self.low_lst, self.high_lst = self.get_dataPath(data_dir, isTraining)
        self.channel = channel
        self.scale_size = scale_size
        self.isTraining = isTraining
        self.name = ""
    
    def __getitem__(self, index):
        low_path = self.low_lst[index]
        self.name = low_path.split("/")[-1]
        high_path = self.high_lst[index]
        simple_transform = transforms.ToTensor()
        
        low = Image.open(low_path)
        high = Image.open(high_path)
        
        if self.channel == 1:
            low = low.convert("L")
            high = high.convert("L")
            norm_transform = transforms.Normalize((0.5,), (0.5,))
        else:
            low = low.convert("RGB")
            high = high.convert("RGB")
            norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        low = transforms.Resize((self.scale_size, self.scale_size))(low)
        high = transforms.Resize((self.scale_size, self.scale_size))(high)
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            low = low.rotate(angel)
            high = high.rotate(angel)
        
        low = norm_transform(simple_transform(low))
        high = norm_transform(simple_transform(high))
        
        return low, high
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.low_lst)
    
    def get_dataPath(self, data_dir, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param data_dir: 存放的文件夹
        :return:
        """
        if isTraining:
            low_dir = os.path.join(data_dir, "train", "images")
            high_dir = os.path.join(data_dir, "train", "labels")
        else:
            low_dir = os.path.join(data_dir, "test", "images")
            high_dir = os.path.join(data_dir, "test", "labels")
        
        low_lst = sorted(list(map(lambda x: os.path.join(low_dir, x), os.listdir(low_dir))))
        high_lst = sorted(list(map(lambda x: os.path.join(high_dir, x), os.listdir(high_dir))))
        
        assert len(low_lst) == len(high_lst)
        
        return low_lst, high_lst
    
    def getFileName(self):
        return self.name


class OCTA_loader(data.Dataset):
    def __init__(self, data_dir, channel=1, scale_size=512, isTraining=True):
        super(OCTA_loader, self).__init__()
        self.low_lst, self.high_lst = self.get_dataPath(data_dir, isTraining)
        self.channel = channel
        self.scale_size = scale_size
        self.isTraining = isTraining
        self.name = ""
    
    def __getitem__(self, index):
        low_path = self.low_lst[index]
        self.name = low_path.split("/")[-1]
        high_path = self.high_lst[index]
        simple_transform = transforms.ToTensor()
        
        low = Image.open(low_path)
        high = Image.open(high_path)
        
        if self.channel == 1:
            low = low.convert("L")
            high = high.convert("L")
            norm_transform = transforms.Normalize((0.5,), (0.5,))
        else:
            low = low.convert("RGB")
            high = high.convert("RGB")
            norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        low = transforms.Resize((self.scale_size, self.scale_size))(low)
        high = transforms.Resize((self.scale_size, self.scale_size))(high)
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            low = low.rotate(angel)
            high = high.rotate(angel)
        
        low = norm_transform(simple_transform(low))
        high = norm_transform(simple_transform(high))
        
        return low, high
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.low_lst)
    
    def get_dataPath(self, data_dir, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param data_dir: 存放的文件夹
        :return:
        """
        if isTraining:
            low_dir = os.path.join(data_dir, "train", "low")
            high_dir = os.path.join(data_dir, "train", "high")
        else:
            low_dir = os.path.join(data_dir, "test", "low")
            high_dir = os.path.join(data_dir, "test", "high")
        
        low_lst = sorted(list(map(lambda x: os.path.join(low_dir, x), os.listdir(low_dir))))
        high_lst = sorted(list(map(lambda x: os.path.join(high_dir, x), os.listdir(high_dir))))
        
        assert len(low_lst) == len(high_lst)
        
        return low_lst, high_lst
    
    def getFileName(self):
        return self.name
