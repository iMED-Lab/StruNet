# -*- coding: utf-8 -*-

from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_psnr(img1, img2, eps=1e-12):
    # assert both img1 and img2 are tensors with shape [batch_size, channels, height, width]
    # assert pixel value range is -1~1 and type is float32
    # return shape: [batch_size,]
    # mse = ((img1.astype(np.float) - img2.astype(np.float)) ** 2).mean()##
    # psnr = 10 * np.log10(255 ** 2 / mse)##
    mse = (((img1 + 1.0) / 2.0 - (img2 + 1.0) / 2.0) ** 2).mean(-1).mean(-1).mean(-1)
    psnr = 10 * torch.log10((1.0 ** 2 + eps) / (mse + eps))
    
    return psnr


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def gaussian(window_size, sigma):  # 定义高斯核##
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):  # 创建窗口##
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 一维窗口##
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # 二维窗口##
    window = nn.Parameter(data=_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)
    return window


def calc_ssim(img1, img2, window, window_size, channel, alpha=1.0, beta=1.0, gamma=1.0, K1=0.01, K2=0.03, L=1.0):
    assert img1.size() == img2.size()
    noise = torch.Tensor(np.random.normal(0, 0.01, img1.size())).cuda(img1.get_device())
    new_img1 = clip_by_tensor((img1 + 1.0) / 2.0 + noise, torch.Tensor(np.zeros(img1.size())).cuda(img1.get_device()), torch.Tensor(np.ones(img1.size())).cuda(img1.get_device()))
    new_img2 = clip_by_tensor((img2 + 1.0) / 2.0 + noise, torch.Tensor(np.zeros(img2.size())).cuda(img2.get_device()), torch.Tensor(np.ones(img2.size())).cuda(img2.get_device()))
    mu1 = F.conv2d(new_img1, window, padding=window_size // 2, groups=channel)  # 图像1的高斯均值##
    mu2 = F.conv2d(new_img2, window, padding=window_size // 2, groups=channel)  # 图像2的高斯均值##

    mu1_sq = mu1.pow(2)  # 图像1的均方值##
    mu2_sq = mu2.pow(2)  # 图像2的均方值##
    mu1_mu2 = mu1 * mu2  # 图像1和图像2的协均方值##

    sigma1_sq = F.conv2d(new_img1 * new_img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # 图像1的方差值##
    sigma2_sq = F.conv2d(new_img2 * new_img2, window, padding=window_size // 2, groups=channel) - mu2_sq  # 图像2的方差值##
    sigma12 = F.conv2d(new_img1 * new_img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # 图像1和图像2的协方差值##
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0
    
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    l = (2 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)
    ssim_map = l.pow(alpha) * c.pow(beta) * s.pow(gamma)

    return ssim_map.mean(-1).mean(-1).mean(-1)
