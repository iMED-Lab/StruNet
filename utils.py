# -*- coding: utf-8 -*-

import os
import visdom
import numpy as np


def mkdir(path):
    # 引入模块
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


# Adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1.0 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Load dataset
def build_dataset(dataset, data_dir, scale_size=512, isTraining=True):
    if dataset == "CT":
        from datasets import CT_loader
        database = CT_loader(data_dir, 1, isTraining)
    elif dataset == "OCT":
        from datasets import OCT_loader
        database = OCT_loader(data_dir, 1, scale_size, isTraining)
    elif dataset == "OCTA":
        from datasets import OCTA_loader
        database = OCTA_loader(data_dir, 1, scale_size, isTraining)
    else:
        raise NotImplementedError('dataset [%s] is not implemented' % dataset)
    
    return database


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """
    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        # 画的第几个数，相当于横坐标
        # 比如("loss", 23) 即loss的第23个点
        self.index = {}
        self.log_text = ""
    
    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        
        return self
    
    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ("loss", 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)
    
    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)
    
    def plot(self, name, y, **kwargs):
        # self.plot("loss", 1.00)
        
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else "append",
                      **kwargs
                      )
        self.index[name] = x + 1
    
    def img(self, name, img_, **kwargs):
        """
        self.img("input_img", t.Tensor(64, 64))
        self.img("input_imgs", t.Tensor(3, 64, 64))
        self.img("input_imgs", t.Tensor(100, 1, 64, 64))
        self.img("input_imgs", t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )
    
    def log(self, info, win="log_text"):
        """
        self.log({"loss": 1, "lr": 0.0001})
        """
        self.log_text += ("[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info))
        self.vis.text(self.log_text, win)
    
    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot, image, log, plot_many等除外
        """
        return getattr(self.vis, name)
