# -*- coding: utf-8 -*-

import argparse


parser = argparse.ArgumentParser(description="DenoiseNet")

parser.add_argument("--gpu_ids", type=str, default="0", help="device")
parser.add_argument("--dataset", type=str, default="CT", choices=["CT", "OCT", "OCTA"], help="dataset")
parser.add_argument("--data_dir", type=str, default=".", help="path to folder for getting dataset")
parser.add_argument("--scale_size", type=int, default=512, help="scale size")
parser.add_argument("--model", type=str, default="vgg19", choices=["vgg16", "vgg19"], help="model")
parser.add_argument("--name", type=str, default="U-Net", help="name of model")

parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=0, help="number of threads")  # 64##
# parser.add_argument("--pin_memory", type=bool, default=True, help="whether to save data in pin memory or not")  ##
parser.add_argument('--pin_memory', action='store_true', help='whether to save data in pin memory or not')
parser.add_argument("--init_lr", type=float, default=0.0005, help="initial learning rate")
parser.add_argument("--power", type=float, default=0.9, help="power")
parser.add_argument("--weight_decay", type=float, default=0.005, help="weight decay")
parser.add_argument("--val_epoch_freq", type=int, default=1, help="frequency of validation at the end of epochs")
parser.add_argument("--load_epoch", type=int, default=0, help="epoch to be loaded")
parser.add_argument("--num_epochs", type=int, default=100, help="train epochs")

# parser.add_argument("--isRes", type=bool, default=True, help="use ResNet block or not")  ##
# parser.add_argument("--isSwinT", type=bool, default=True, help="use swin transformer block or not")  ##
parser.add_argument('--isRes', action='store_true', help='use ResNet block or not')
parser.add_argument('--isSwinT', action='store_true', help='use swin transformer block or not')

parser.add_argument("--window_size", type=int, default=2, help="window size")
parser.add_argument("--weight_l1", type=float, default=1.0, help="weight of L1 loss")
parser.add_argument("--weight_percept", type=float, default=0.05, help="weight of perceptual loss")  # 0.1##
parser.add_argument("--weight_rank", type=float, default=0.0001, help="weight of rank regularization")  # 0.002##
parser.add_argument("--eps", type=float, default=0.005, help="threshold of rank regularization")

parser.add_argument("--logs_dir", type=str, default="logs", help="path to folder for saving logs")
parser.add_argument("--models_dir", type=str, default="models", help="path to folder for saving enhancement/degradation models")
parser.add_argument("--results_dir", type=str, default="results", help="path to folder for saving enhancement/degradation results")
parser.add_argument("--load_model", type=str, default="", help="model to be loaded")  # best-best.pth##

args = parser.parse_args()
