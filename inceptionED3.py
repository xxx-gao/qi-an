
from .fusion import *
from .dense import *
from .deconv import FastDeconv
from .IncceptionNext import *

from einops import rearrange
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(in_features),
                        nn.PReLU(),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(out_features, out_features, 3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class PALayer(nn.Module):
    def __init__(self, nc, number):
        super(PALayer, self).__init__()
        self.conv = nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(number)
        self.at = nn.Sequential(
            nn.Conv2d(number, 1, 1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.at(y)
        return x * y

class CALayer(nn.Module):
    def __init__(self, nc, number):
        super(CALayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(nc)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.at = nn.Sequential(
            nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.avg_pool(y)
        y = self.at(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, nc, number=4):
        super(DehazeBlock, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nc)
        self.act1 = nn.PReLU()

        self.calayer = CALayer(nc, number)
        self.palayer = PALayer(nc, number)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act1(self.conv1(res))
        res = res + x
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class YCrCb(nn.Module):
    def __init__(self):
        super(YCrCb, self).__init__()
        self.y_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).cuda()
        self.cb_weight = torch.tensor([-0.169, -0.331, 0.500]).view(1, 3, 1, 1).cuda()
        self.cr_weight = torch.tensor([0.500, -0.419, -0.081]).view(1, 3, 1, 1).cuda()

    def forward(self, img):
        y = F.conv2d(img, self.y_weight)
        cb = F.conv2d(img, self.cb_weight) + 0.5
        cr = F.conv2d(img, self.cr_weight) + 0.5

        return torch.cat([cr, cb], dim=1)

class  net(nn.Module):
    def __init__(self, input_nc=3,in_features=32, n_residual_att=6, n_residual_blocks=6):
        super(net, self).__init__()

        self.deconv = FastDeconv(input_nc, input_nc, 3, padding=1)

        att = [nn.ReflectionPad2d(3),
               nn.Conv2d(input_nc, in_features // 2, 7),
               nn.BatchNorm2d(in_features // 2),
               nn.PReLU()]
        for _ in range(n_residual_att):
            att += [DehazeBlock(in_features // 2)]  # 16

        att += [nn.ReflectionPad2d(3),
                nn.Conv2d(in_features // 2, 1, 7),
                nn.Sigmoid()]

        self.att = nn.Sequential(*att)

        self.ycrcb = YCrCb()

        self.cr = crcb()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.rb1 = ResidualBlock(64, 64)


        self.rb1_1 = ResidualBlock(in_features=64, out_features=64)

        self.down1_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )




        self.rb2_1 = ResidualBlock(in_features=128, out_features=128)

        self.down2_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )


        self.rb3_1 = ResidualBlock(in_features=256, out_features=256)
        self.db2 = DehazeBlock(256)
        self.db3 = MetaNeXtBlock(256)
        self.db4 = DehazeBlock(256)
        self.rb4_1 = ResidualBlock(in_features=256, out_features=256)



        self.mff2 = MFF(128)
        self.mff3 = MFF(256)
        self.cga1 = CGA1Fusion(256, reduction=8)
        self.cga2 = CGA1Fusion(128, reduction=8)



        self.up1_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.rb5_1 = ResidualBlock(in_features=128, out_features=128)



        self.up2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.rb6_1 = ResidualBlock(in_features=64, out_features=64)


        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0, bias=True),
            nn.Tanh()
        )

        self.db1 = DehazeBlock(64)


        self.fusion1 = Fusion(96,64)
        self.db1_1 = DehazeBlock(64)
        self.db1_2 = DehazeBlock(64)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.25)
        self.gamma_max = 0.1
        self.gamma_min = 0.01
    def forward(self,x, m_gt=None, l_mask=0.00001):
        beta = 1.0
        if l_mask >= self.gamma_max:
            beta = 1.0
        elif l_mask >= self.gamma_min:
            beta = (l_mask - self.gamma_min) / (self.gamma_max - self.gamma_min)
        else:
            beta = 0.0

        x_deconv = self.deconv(x)

        m_g = self.att(x_deconv)

        if m_gt is None:
            m = m_g
        else:

            m = 0 * m_gt + (1 - 0) * m_g


        x1 = self.ycrcb(x)
        x1 = self.cr(x1)
        x2 = self.conv(x)
        x2 = self.rb1(x2)
        x2 = self.db1(x2)
        xin = torch.cat([x1, x2], 1)#128
        xin = self.fusion1(xin)
        x_inp = self.alpha * m * xin + (1 - self.alpha) * xin
        x_inp = self.db1_1(x_inp)
        x_inp = self.db1_1(x_inp)
        xdown1= self.rb1_1(x_inp)
        xdown2 = self.down1_2(xdown1)

        x3 = self.rb2_1(xdown2)
        xdown3 = self.down2_2(x3)


        mff_fa2 = self.mff2(xdown1, xdown2, xdown3)
        mff_fa3 = self.mff3(xdown1, xdown2, xdown3)


        x3 = self.rb3_1(xdown3)
        x3 = self.db2(x3)
        x3 = self.db3(x3)
        x3 = self.db4(x3)
        x3 = self.rb4_1(x3)
        x3 = self.cga1(x3,mff_fa3)
        x3 = self.up1_2(x3)
        x3 = self.rb5_1(x3)


        x_mix2 = self.cga2(x3,mff_fa2)
        x4 = self.up2_2(x_mix2)

        x4 = self.rb6_1(x4)
        x4 = self.fusion(x4)


        return x4


class Fusion(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_features, 7, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.fusion(input)
        return out

class crcb(nn.Module):
    def __init__(self):
        super(crcb, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.denseblock = RDB(32,32,3)
        self.dbc = Attention(32, 1, False)

    def forward(self, x):
        x3 = self.conv2(x)
        x3 = self.denseblock(x3)

        x3 = self.dbc(x3)

        return x3



class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))


if __name__ == '__main__':


    print('==> Building model..')

    model = net().cuda()



