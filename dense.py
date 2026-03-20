import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels,  growth_rate=64,num_layers=3,kernel_size=5):

        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
                     nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size // 2),
                               groups=in_channels, bias=True),nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
                                  )
        self.relu = nn.PReLU()

    def forward(self, x):
        x = torch.cat([x, self.relu(self.conv(x))], 1)

        return x



class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):#64，64，3
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)  #k=1

    def forward(self, x, lrl=True):
        if lrl:
            return x + self.lff(self.layers(x))  # local residual learning
