# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from options import args
from utils import build_dataset, mkdir
from evaluation import create_window, calc_psnr, calc_ssim


# Use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
np.random.seed(0)

# Load dataset
testbase = build_dataset(args.dataset, args.data_dir, args.scale_size, isTraining=False)
test_loader = DataLoader(testbase, batch_size=1, shuffle=False, num_workers=0, pin_memory=args.pin_memory)  # args.num_workers##

# Load model
Denoise_Net = torch.load(os.path.join(args.models_dir, args.dataset, args.name, args.load_model), map_location=device)

if isinstance(Denoise_Net, nn.DataParallel):
    Denoise_Net = Denoise_Net.module

Denoise_Net.eval()

mkdir(os.path.join(args.results_dir, args.dataset, "Low"))
mkdir(os.path.join(args.results_dir, args.dataset, "High"))
mkdir(os.path.join(args.results_dir, args.dataset, args.name))
window = create_window(11, 1).to(device)
psnr_lst = []
ssim_lst = []
time_lst = []

# Start testing
print("Start testing...")
with torch.no_grad():
    for step, (image, label) in enumerate(test_loader):
        image = image.to(device)
        label = label.to(device)
        
        assert image.size() == label.size()
        image_arr = (image.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0
        label_arr = (label.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0
        
        start_t = time.time()
        pred_high = Denoise_Net(image)
        end_t = time.time()
        print("cost time: %f s" % (end_t - start_t))
        time_lst.append(end_t - start_t)
        
        psnr = calc_psnr(pred_high, label)
        ssim = calc_ssim(pred_high, label, window, 11, 1)
        psnr_lst.append(psnr.mean(-1).item())
        ssim_lst.append(ssim.mean(-1).item())
        
        pred_high_arr = (pred_high.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0
        
        name = test_loader.dataset.getFileName()
        cv2.imwrite(os.path.join(args.results_dir, args.dataset, "Low", name), image_arr.astype(np.uint8))
        cv2.imwrite(os.path.join(args.results_dir, args.dataset, args.name, name), pred_high_arr.astype(np.uint8))
        cv2.imwrite(os.path.join(args.results_dir, args.dataset, "High", name), label_arr.astype(np.uint8))
        print(name + " done. ")

assert len(psnr_lst) == len(ssim_lst)
if len(psnr_lst) > 0:
    psnr_arr = np.array(psnr_lst)
    ssim_arr = np.array(ssim_lst)
    print("PSNR: " + str(psnr_arr.mean()) + " ± " + str(psnr_arr.std()))
    print("SSIM: " + str(ssim_arr.mean()) + " ± " + str(ssim_arr.std()))
print("Testing finished. ")
time_arr = np.array(time_lst)
print("Time: %f ± %f s" % (time_arr.mean(), time_arr.std()))
