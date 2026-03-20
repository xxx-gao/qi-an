import torch
import torch.nn as nn

class MFF(nn.Module):

    def __init__(self, feature_channel,in_channels=64,height=3, reduction=8, bias=False):
        super(MFF, self).__init__()


        if feature_channel == 128:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel*2, feature_channel, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))

            self.feature_channel = feature_channel


        else:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 4, feature_channel, kernel_size=5, stride=4, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))

            self.feature_channel = feature_channel
        self.height = height
        d = max(int(feature_channel / reduction), 4)
        self.conv_du = nn.Sequential(nn.Conv2d(feature_channel, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, feature_channel, kernel_size=1, stride=1, bias=bias))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, bias=True),
                                  nn.Conv2d(feature_channel, feature_channel, kernel_size=5,
                                            padding=(5// 2),
                                            groups=feature_channel, bias=True),

                                  )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2, f3):
        batch_size = f1.shape[0]

        feature1 = self.conv1(f1).unsqueeze(dim=1)
        feature2 = self.conv2(f2).unsqueeze(dim=1)
        feature3 = self.conv3(f3).unsqueeze(dim=1)
        feature12 = torch.cat([feature1, feature2],dim=1)
        feature123 = torch.cat([feature12, feature3],dim=1)
        fea_U = torch.sum(feature123, dim=1)
        fea_U = self.conv(fea_U)

        u = self.avg_pool(fea_U)

        feats_Z = self.conv_du(u)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]

        attention_vectors = torch.cat(attention_vectors, dim=1)

        attention_vectors = attention_vectors.view(batch_size, self.height, self.feature_channel, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feature123 * attention_vectors, dim=1)

        return  feats_V


from torch import nn

import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1×1
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  #
        pattn1 = pattn1.unsqueeze(dim=2)  #
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=1.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)




class CGA1Fusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGA1Fusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.cm = CCM(dim)
    def forward(self, x, y):
        initial = x + y
        initial = self.cm(initial)
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

