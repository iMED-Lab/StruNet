# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from timm.models.layers import trunc_normal_
from swin_transformer import PatchEmbed, PatchMerging, BasicLayer


def set_parameter_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=False)
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.up(x)
        
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        
        return self.relu(out + residual)


class VGG(nn.Module):
    def __init__(self, model="vgg19"):
        super(VGG, self).__init__()
        assert model == "vgg16" or model == "vgg19"
        if model == "vgg16":
            net = models.vgg16(pretrained=True)
            net.eval()
            set_parameter_requires_grad(net.features, requires_grad=False)
            self.layer1 = net.features[:4]
            self.layer2 = net.features[5:9]
            self.layer3 = net.features[10:16]
            self.layer4 = net.features[17:23]
            self.layer5 = net.features[24:30]
        else:
            net = models.vgg19(pretrained=True)
            net.eval()
            set_parameter_requires_grad(net.features, requires_grad=False)
            self.layer1 = net.features[:4]
            self.layer2 = net.features[5:9]
            self.layer3 = net.features[10:18]
            self.layer4 = net.features[19:27]
            self.layer5 = net.features[28:36]
    
    def forward(self, x):
        relu1 = self.layer1(x)
        x = F.max_pool2d(relu1, kernel_size=2, stride=2)
        
        relu2 = self.layer2(x)
        x = F.max_pool2d(relu2, kernel_size=2, stride=2)
        
        relu3 = self.layer3(x)
        x = F.max_pool2d(relu3, kernel_size=2, stride=2)
        
        relu4 = self.layer4(x)
        x = F.max_pool2d(relu4, kernel_size=2, stride=2)
        
        relu5 = self.layer5(x)
        # x = F.max_pool2d(relu5, kernel_size=2, stride=2)
        
        return relu1, relu2, relu3, relu4, relu5  # , x##


class Decoder(nn.Module):
    def __init__(self, output_nc=1, channels=[64, 128, 256, 512, 1024], p=0.1, window_size=4, isRes=True, isSwinT=True):  # p=0.##
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p=p, inplace=True)
        
        self.Up5 = up_conv(ch_in=channels[4], ch_out=channels[3])
        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2])
        self.Up3 = up_conv(ch_in=channels[2], ch_out=channels[1])
        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[0])
        self.isSwinT = isSwinT
        
        if isRes:
            self.Up_conv5 = res_conv_block(ch_in=channels[4], ch_out=channels[3])
            self.Up_conv4 = res_conv_block(ch_in=channels[3], ch_out=channels[2])
            self.Up_conv3 = res_conv_block(ch_in=channels[2], ch_out=channels[1])
            self.Up_conv2 = res_conv_block(ch_in=channels[1], ch_out=channels[0])
        else:
            self.Up_conv5 = conv_block(ch_in=channels[4], ch_out=channels[3])
            self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2])
            self.Up_conv3 = conv_block(ch_in=channels[2], ch_out=channels[1])
            self.Up_conv2 = conv_block(ch_in=channels[1], ch_out=channels[0])
        
        if self.isSwinT:
            dpr = [x.item() for x in torch.linspace(0, 0.1, 22)]
            self.SwinT5 = BasicLayer(dim=channels[3], input_resolution=(64, 64), depth=18, num_heads=16, window_size=window_size, mlp_ratio=4,
                                       qkv_bias=True, qk_scale=None, drop=0, attn_drop=0, drop_path=dpr[4:22], norm_layer=nn.LayerNorm,
                                       downsample=None, use_checkpoint=False)
            self.SwinT4 = BasicLayer(dim=channels[2], input_resolution=(128, 128), depth=2, num_heads=8, window_size=window_size, mlp_ratio=4,
                                       qkv_bias=True, qk_scale=None, drop=0, attn_drop=0, drop_path=dpr[2:4], norm_layer=nn.LayerNorm,
                                       downsample=None, use_checkpoint=False)
            self.SwinT3 = BasicLayer(dim=channels[1], input_resolution=(256, 256), depth=2, num_heads=4, window_size=window_size, mlp_ratio=4,
                                       qkv_bias=True, qk_scale=None, drop=0, attn_drop=0, drop_path=dpr[0:2], norm_layer=nn.LayerNorm,
                                       downsample=None, use_checkpoint=False)
            # self.Up_conv2 = nn.Conv2d(channels[1], channels[0], kernel_size=2, stride=2)##
        
        self.Conv_1x1 = nn.Conv2d(channels[0], output_nc, kernel_size=1)
    
    def forward(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)  # [B, 1024, 32, 32] --> [B, 512, 64, 64]##
        if self.isSwinT:
            B, C, H, W = d5.size()
            d5 = d5 + self.SwinT5(d5.view(B, C, H * W).transpose(1, 2))  # [B, 64*64, 512] --> [B, 512, 64, 64]##
        d5 = torch.cat((x4, d5), dim=1)  # [B, 512, 64, 64] * 2 --> [B, 1024, 64, 64]##
        d5 = self.Up_conv5(d5)  # [B, 1024, 64, 64] --> [B, 512, 64, 64]##
        # if self.isSwinT:##
        #     B, C, H, W = d5.size()##
        #     d5 = self.SwinT5(d5.view(B, C, H * W).transpose(1, 2))##
        d5 = self.dropout(d5)
        
        d4 = self.Up4(d5)  # [B, 512, 64, 64] --> [B, 256, 128, 128]##
        if self.isSwinT:
            B, C, H, W = d4.size()
            d4 = d4 + self.SwinT4(d4.view(B, C, H * W).transpose(1, 2))  # [B, 128*128, 256] --> [B, 256, 128, 128]##
        d4 = torch.cat((x3, d4), dim=1)  # [B, 256, 128, 128] * 2 --> [B, 512, 128, 128]##
        d4 = self.Up_conv4(d4)  # [B, 512, 128, 128] --> [B, 256, 128, 128]##
        # if self.isSwinT:##
        #     B, C, H, W = d4.size()##
        #     d4 = self.SwinT4(d4.view(B, C, H * W).transpose(1, 2))##
        d4 = self.dropout(d4)
        
        d3 = self.Up3(d4)  # [B, 256, 128, 128] --> [B, 128, 256, 256]##
        if self.isSwinT:
            B, C, H, W = d3.size()
            d3 = d3 + self.SwinT3(d3.view(B, C, H * W).transpose(1, 2))  # [B, 256*256, 128] --> [B, 128, 256, 256]##
        d3 = torch.cat((x2, d3), dim=1)  # [B, 128, 256, 256] * 2 --> [B, 256, 256, 256]##
        d3 = self.Up_conv3(d3)  # [B, 256, 256, 256] --> [B, 128, 256, 256]##
        # if self.isSwinT:##
        #     B, C, H, W = d3.size()##
        #     d3 = self.SwinT3(d3.view(B, C, H * W).transpose(1, 2))##
        d3 = self.dropout(d3)
        
        d2 = self.Up2(d3)  # [B, 128, 256, 256] --> [B, 64, 512, 512]##
        d2 = torch.cat((x1, d2), dim=1)  # [B, 64, 512, 512] * 2 --> [B, 128, 512, 512]##
        d2 = self.Up_conv2(d2)  # [B, 128, 512, 512] --> [B, 64, 512, 512]##
        
        d1 = self.Conv_1x1(d2)  # [B, 64, 512, 512] --> [B, 1, 512, 512]##
        out = nn.Tanh()(d1)
        
        return out


class BasicEncoder(nn.Module):
    def __init__(self, input_nc=1, channels=[64, 128, 256, 512, 1024], isRes=True):
        super(BasicEncoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)
        
        if isRes:
            self.Conv1 = res_conv_block(ch_in=input_nc, ch_out=channels[0])
            self.Conv2 = res_conv_block(ch_in=channels[0], ch_out=channels[1])
            self.Conv3 = res_conv_block(ch_in=channels[1], ch_out=channels[2])
            self.Conv4 = res_conv_block(ch_in=channels[2], ch_out=channels[3])
            self.Conv5 = res_conv_block(ch_in=channels[3], ch_out=channels[4])
        else:
            self.Conv1 = conv_block(ch_in=input_nc, ch_out=channels[0])
            self.Conv2 = conv_block(ch_in=channels[0], ch_out=channels[1])
            self.Conv3 = conv_block(ch_in=channels[1], ch_out=channels[2])
            self.Conv4 = conv_block(ch_in=channels[2], ch_out=channels[3])
            self.Conv5 = conv_block(ch_in=channels[3], ch_out=channels[4])
    
    def forward(self, x):
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        return x1, x2, x3, x4, x5


class SwinTransformerEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=2, in_chans=1,
                 embed_dim=128, depths=[2, 2, 18], num_heads=[4, 8, 16],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, isRes=True, isSwinT=True, **kwargs):  # conv_type="", "basic", "res"##
        super(SwinTransformerEncoder, self).__init__()
        
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.isSwinT = isSwinT  # ##
        
        self.conv_layers = nn.ModuleList()
        if isRes:
            for i in range(5):
                if i == 0:
                    self.conv_layers.append(res_conv_block(ch_in=in_chans, ch_out=embed_dim//2))
                else:
                    self.conv_layers.append(res_conv_block(ch_in=int(embed_dim*(2**(i-2))), ch_out=int(embed_dim*(2**(i-1)))))
        else:
            for i in range(5):
                if i == 0:
                    self.conv_layers.append(conv_block(ch_in=in_chans, ch_out=embed_dim//2))
                else:
                    self.conv_layers.append(conv_block(ch_in=int(embed_dim*(2**(i-2))), ch_out=int(embed_dim*(2**(i-1)))))
        self.Maxpool = nn.MaxPool2d(2)
        
        # split image into non-overlapping patches
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=in_chans, embed_dim=embed_dim//2,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed2 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim//2, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches1 = self.patch_embed1.num_patches
        num_patches2 = self.patch_embed2.num_patches
        patches_resolution = self.patch_embed2.patches_resolution
        self.patches_resolution = patches_resolution
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, num_patches1, embed_dim//2))
            trunc_normal_(self.absolute_pos_embed1, std=.02)
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, num_patches2, embed_dim))
            trunc_normal_(self.absolute_pos_embed2, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging,  # if (i_layer < self.num_layers - 1) else None##
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward(self, x):
        B, C, H, W = x.size()
        x_res1 = self.conv_layers[0](x)  # [B, 1, 512, 512] --> [B, 64, 512, 512]##
        if self.isSwinT:
            x = self.patch_embed1(x)
            if self.ape:
                x = x + self.absolute_pos_embed1
            x = self.pos_drop(x)  # [B, 512*512, 64]##
            x_res1 = x.transpose(1, 2).view(B, self.embed_dim//2, H, W) + x_res1  # [B, 64, 512, 512]##
            # x_res1 = self.conv_layers[0](x_res1)  ##
        
        x_res2 = self.conv_layers[1](x_res1)  # [B, 64, 512, 512] --> [B, 128, 512, 512]##
        x_res2 = self.Maxpool(x_res2)  # [B, 128, 512, 512] --> [B, 128, 256, 256]##
        if self.isSwinT:
            x = self.patch_embed2(x_res1)
            if self.ape:
                x = x + self.absolute_pos_embed2
            x = self.pos_drop(x)  # [B, 256*256, 128]##
            x_res2 = x.transpose(1, 2).view(B, self.embed_dim, H//self.patch_size, W//self.patch_size) + x_res2  # [B, 128, 256, 256]##
            # x2 = self.conv_layers[1](x2)  ##
            # x = x2.view(B, self.embed_dim, -1).transpose(1, 2)  ##
        
        x_lst = [x_res1, x_res2]  # [B, 64, 512, 512] | [B, 128, 256, 256]##
        i = 2  # ##
        for layer in self.layers:
            x_res = self.conv_layers[i](x_lst[-1])  # [B, 128, 256, 256] --> [B, 256, 256, 256]##
            x_res = self.Maxpool(x_res)  # [B, 256, 256, 256] --> [B, 256, 128, 128]##
            if self.isSwinT:
                x_res = layer(x) + x_res  # [B, 256*256, 128] --> [B, 256, 128, 128] | [B, 128*128, 256] --> [B, 512, 64, 64] | [B, 64*64, 512] --> [B, 1024, 32, 32]##
                #     y = self.conv_layers[i](y)  ##
                B, C, H, W = x_res.size()  # ##
                x = x_res.view(B, C, -1).transpose(1, 2)  # ##
            x_lst.append(x_res)
            i += 1
        
        # x = self.norm(x)  B L C##
        return x_lst  # [B, 64, 512, 512], [B, 128, 256, 256], [B, 256, 128, 128], [B, 512, 64, 64], [B, 1024, 32, 32]##
    
    def flops(self):
        flops = 0
        flops += self.patch_embed1.flops()
        flops += self.patch_embed2.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops


class EncoderDecoder(nn.Module):
    def __init__(self, window_size=4, isRes=True, isSwinT=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = SwinTransformerEncoder(window_size=window_size, isRes=isRes, isSwinT=isSwinT)
        self.decoder = Decoder(window_size=window_size, isRes=isRes, isSwinT=isSwinT)
    
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        y = self.decoder(x1, x2, x3, x4, x5)
        
        return y
