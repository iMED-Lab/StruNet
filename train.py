# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args
from networks import VGG, EncoderDecoder
from losses import percept_loss
from optimizer import build_optimizer  # ##
from config import _C  # ##
from utils import build_dataset, mkdir, get_lr, adjust_lr, Visualizer
from evaluation import create_window, calc_psnr, calc_ssim


# Use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
np.random.seed(0)

# Load dataset
trainbase = build_dataset(args.dataset, args.data_dir, args.scale_size, isTraining=True)
train_loader = DataLoader(trainbase, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
valbase = build_dataset(args.dataset, args.data_dir, args.scale_size, isTraining=False)
val_loader = DataLoader(valbase, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

# Build model
Denoise_Net = EncoderDecoder(window_size=args.window_size, isRes=args.isRes, isSwinT=args.isSwinT).to(device)
if args.load_model != "":
    Denoise_Net = torch.load(os.path.join(args.models_dir, args.dataset, args.name, args.load_model), map_location=device)
Denoise_Net = nn.DataParallel(Denoise_Net)

fe_layer = VGG(args.model).to(device)
fe_layer = nn.DataParallel(fe_layer)

# Set up optimization
# optimizer = optim.Adam(Denoise_Net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)  ##
optimizer = build_optimizer(_C, Denoise_Net)  # ##
criterion = nn.L1Loss()

# Create logs
mkdir(os.path.join(args.models_dir, args.dataset, args.name))
viz = Visualizer(env=args.name + "_" + args.dataset)
writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, args.name))
dt_size = len(train_loader.dataset)
checkpoint_path = os.path.join(args.models_dir, args.dataset, "{net}", "{epoch}-{loss}.pth")

f = open(os.path.join(args.logs_dir, args.dataset, args.name, "options.txt"), "w")
f.write(str(args))
f.write("\n")
f.write(str(_C))
# print(_C, f) <_io.TextIOWrapper name='logs/CT/STUNet/options.txt' mode='w' encoding='UTF-8'>##
f.close()

f = open(os.path.join(args.logs_dir, args.dataset, args.name, "network.txt"), "w")
f.write(str(Denoise_Net))
f.close()

# Start training
best_loss = float("inf")
best_epoch = 0
window = create_window(11, 1).to(device)
print("Start training...")
for epoch in range(args.load_epoch, args.num_epochs):
    epoch_loss = 0.0
    step = 0
    Denoise_Net.train(mode=True)
    # fe_layer.train(mode=True)##
    for (low, high) in train_loader:
        step += 1
        img_low = low.to(device)
        img_high = high.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        pred_high = Denoise_Net(img_low)
        loss = percept_loss(pred_high, img_high, fe_layer, criterion=criterion, eps=args.eps,
                            weight_rank=args.weight_rank, weight_l1=args.weight_l1, weight_percept=args.weight_percept)  # ##
        # loss = criterion(pred_high, img_high)  ##
        
        viz.img(name="in_low", img_=(img_low[0, :, :, :]+1.0)/2.0)
        viz.img(name="in_high", img_=(img_high[0, :, :, :]+1.0)/2.0)
        viz.img(name="out_high", img_=(pred_high[0, :, :, :]+1.0)/2.0)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Loss of current batch
        niter = epoch * len(train_loader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("Epoch %d, Step %d / %d, Train loss: %0.4f"
              % (epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
        with open(os.path.join(args.logs_dir, args.dataset, args.name, "loss_log.txt"), "a") as log_file:  # Save the message
            log_file.write("Epoch %d, Step %d / %d, Train loss: %0.4f\n"
                           % (epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())
    
        # Write current lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f\nlearning rate: %f" % (epoch, epoch_loss / step, current_lr))
    with open(os.path.join(args.logs_dir, args.dataset, args.name, "loss_log.txt"), "a") as log_file:  # Save the message
        log_file.write("epoch %d loss: %0.4f\nlearning rate: %f\n" % (epoch, epoch_loss / step, current_lr))
    
    adjust_lr(optimizer, args.init_lr, epoch, args.num_epochs, power=args.power)
    
    if (epoch + 1) % args.val_epoch_freq == 0 or epoch + 1 == args.num_epochs:
        Denoise_Net.eval()
        # fe_layer.eval()##
        loss_lst = []
        psnr_lst = []
        ssim_lst = []
        with torch.no_grad():
            for (low, high) in val_loader:
                img_low = low.to(device)
                img_high = high.to(device)
                
                pred_high = Denoise_Net(img_low)
                
                viz.img(name="te_low", img_=(img_low[0, :, :, :]+1.0)/2.0)
                viz.img(name="te_high", img_=(img_high[0, :, :, :]+1.0)/2.0)
                viz.img(name="pred_high", img_=(pred_high[0, :, :, :]+1.0)/2.0)
                
                loss = percept_loss(pred_high, img_high, fe_layer, criterion=criterion, eps=args.eps,
                                    weight_rank=args.weight_rank, weight_l1=args.weight_l1, weight_percept=args.weight_percept)  # ##
                # loss = criterion(pred_high, img_high)  ##
                psnr = calc_psnr(pred_high, img_high)
                ssim = calc_ssim(pred_high, img_high, window, 11, 1)
                loss_lst.append(loss.item())
                psnr_lst.append(psnr.mean(-1).item())
                ssim_lst.append(ssim.mean(-1).item())
        
        loss_arr = np.array(loss_lst)
        psnr_arr = np.array(psnr_lst)
        ssim_arr = np.array(ssim_lst)
        writer.add_scalars("val_loss", {"val_loss": loss_arr.mean()}, epoch + 1)
        viz.plot("val loss", loss_arr.mean())
        print("Val Loss: " + str(loss_arr.mean()) + " ± " + str(loss_arr.std()))
        writer.add_scalars("val_psnr", {"val_psnr": psnr_arr.mean()}, epoch + 1)
        viz.plot("val psnr", psnr_arr.mean())
        print("Val PSNR: " + str(psnr_arr.mean()) + " ± " + str(psnr_arr.std()))
        writer.add_scalars("val_ssim", {"val_ssim": ssim_arr.mean()}, epoch + 1)
        viz.plot("val ssim", ssim_arr.mean())
        print("Val SSIM: " + str(ssim_arr.mean()) + " ± " + str(ssim_arr.std()))
        
        torch.save(Denoise_Net, checkpoint_path.format(net=args.name, epoch=epoch+1, loss=loss_arr.mean()))
        print("Model saved - Epoch: %d, Val Loss: %f, Val PSNR: %f, Val SSIM: %f"
              % (epoch + 1, loss_arr.mean(), psnr_arr.mean(), ssim_arr.mean()))
        with open(os.path.join(args.logs_dir, args.dataset, args.name, "loss_log.txt"), "a") as log_file:  # Save the message
            log_file.write("Model saved - Epoch: %d, Val Loss: %f, Val PSNR: %f, Val SSIM: %f\n"
                           % (epoch + 1, loss_arr.mean(), psnr_arr.mean(), ssim_arr.mean()))
        
        if best_loss >= loss_arr.mean():
            best_loss = loss_arr.mean()
            best_epoch = epoch + 1
            torch.save(Denoise_Net, checkpoint_path.format(net=args.name, epoch="best", loss="best"))
        print("Best model saved - Epoch: %d, Val Loss: %f" % (best_epoch, best_loss))
        with open(os.path.join(args.logs_dir, args.dataset, args.name, "loss_log.txt"), "a") as log_file:  # Save the message
            log_file.write("Best model saved - Epoch: %d, Val Loss: %f\n" % (best_epoch, best_loss))
print("Training finished. ")
