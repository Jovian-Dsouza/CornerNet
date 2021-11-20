import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# from lib.cpool import TopPool, BottomPool, LeftPool, RightPool
from cornerpool import TopPool, BottomPool, LeftPool, RightPool

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, 
                              padding=(kernel_size-1)//2,
                              stride=stride,
                              bias=not batchnorm,
                              )
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv_bn = Conv(in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            batchnorm=True,
                            )
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size-1)//2
                               )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, inputs):
        x = self.conv_bn(inputs)
        x = self.bn2(self.conv2(x))
        return self.relu(x + self.downsample(inputs))

class pool(nn.Module):
  def __init__(self, dim, pool1, pool2):
    super(pool, self).__init__()
    self.p1_conv1 = Conv(dim, 128, kernel_size=3)
    self.p2_conv1 = Conv(dim, 128, kernel_size=3)

    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)

    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)

    self.conv2 = Conv(dim, dim, kernel_size=3)

    self.pool1 = pool1()
    self.pool2 = pool2()

  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))

    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
    bn1 = self.bn1(self.conv1(x))

    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out


def make_res_layer(in_channels, out_channels, kernel_size, modules, stride=1):
    layers = [Residual(in_channels, out_channels, kernel_size=kernel_size, stride=stride)]
    layers += [Residual(out_channels, out_channels, kernel_size=kernel_size) for _ in range(modules-1)]
    return nn.Sequential(*layers)

def make_reverse_res_layer(in_channels, out_channels, kernel_size, modules):
    layers = [Residual(in_channels, in_channels, kernel_size) for _ in range(modules-1)]
    layers.append(Residual(in_channels, out_channels, kernel_size))
    return nn.Sequential(*layers)

# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(Conv(cnv_dim, curr_dim, kernel_size=3, batchnorm=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))

class HourGlass(nn.Module):
    def __init__(self, channels, modules):
        super().__init__()

        assert len(channels) == len(modules), '\n\nchannels list and modules list should be of same len\n'

        n = len(channels) - 1
        curr_modules = modules[0]
        in_channels = channels[0]
        out_channls = channels[1]

        self.layer1 = make_res_layer(in_channels, in_channels, 3, curr_modules)
        self.layer2 = make_res_layer(in_channels, out_channls, 3, curr_modules, stride=2)
        if n == 1:
            next_modules = modules[1]
            self.layer3 = make_res_layer(out_channls, out_channls, 3, next_modules)
        else:
            self.layer3 = HourGlass(channels[1:], modules[1:])
        self.layer4 = make_res_layer(out_channls, in_channels, 3, curr_modules)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.layer1(x) # res
        x = self.layer2(x) # res with stride=2
        x = self.layer3(x) # hourglass
        x = self.layer4(x) # res
        up2 = self.up(x) # upsample
        return up1 + up2

class CornerNet(pl.LightningModule):
    def __init__(self, nstack, channels, modules, num_classes, cnv_dim=256, **kwargs):
        super().__init__()
        # self.save_hyperparameters()

        self.nstack = nstack
        curr_dim = channels[0]

        self.pre = nn.Sequential(
            Conv(3, 128, kernel_size=7, stride=2),
            Residual(128, curr_dim, kernel_size=3, stride=2)
        )
        self.hg = nn.ModuleList([HourGlass(channels, modules) for _ in range(nstack)])
        self.conv = nn.ModuleList([Conv(curr_dim, cnv_dim, kernel_size=3) for _ in range(nstack)]) 

        self.convs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, kernel_size=1, bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])
        self.inters_skip = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, kernel_size=1, bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])
        self.inters = nn.ModuleList([Residual(curr_dim, curr_dim, kernel_size=3) for _ in range(nstack)])
        
        self.relu = nn.ReLU(inplace=True)

        self.pool_tl = nn.ModuleList([pool(cnv_dim, TopPool, LeftPool) for _ in range(nstack)])
        self.pool_br = nn.ModuleList([pool(cnv_dim, BottomPool, RightPool) for _ in range(nstack)])

        # heatmap layers
        self.hmap_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
        self.hmap_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])

        # embedding layers
        self.embd_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
        self.embd_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])

        for hmap_tl, hmap_br in zip(self.hmap_tl, self.hmap_br):
            hmap_tl[-1].bias.data.fill_(-2.19)
            hmap_br[-1].bias.data.fill_(-2.19)

        # regression layers
        self.regs_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.regs_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])


    def forward(self, inputs):
        inter = self.pre(inputs)

        outs = []
        for i in range(self.nstack):
            x = self.hg[i](inter)
            cnv = self.conv[i](x)

            if self.training or i == self.nstack-1:
                cnv_tl = self.pool_tl[i](cnv)
                cnv_br = self.pool_br[i](cnv)

                hmap_tl, hmap_br = self.hmap_tl[i](cnv_tl), self.hmap_br[i](cnv_br)
                embd_tl, embd_br = self.embd_tl[i](cnv_tl), self.embd_br[i](cnv_br)
                regs_tl, regs_br = self.regs_tl[i](cnv_tl), self.regs_br[i](cnv_br)

                outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])

            if i < self.nstack-1:
                inter = self.inters_skip[i](inter) + self.convs_[i](cnv)
                inter = self.relu(inter)
                inter = self.inters[i](inter)
            
        return outs

get_hourglass = {
    'large_hourglass': {
                        'nstack':2,
                        'channels': [256, 256, 384, 384, 384, 512],
                        'modules': [2, 2, 2, 2, 2, 4]
                        },
    'small_hourglass': {
                        'nstack':1,
                        'channels': [256, 256, 384, 384, 384, 512],
                        'modules': [2, 2, 2, 2, 2, 4]
                        },
    'tiny_hourglass': {
                        'nstack':1,
                        'channels': [256, 128, 256, 256, 256, 384],
                        'modules': [2, 2, 2, 2, 2, 4]
                        }
}
class CornerNetPL(CornerNet):
    def __init__(self, model_type, num_classes, cnv_dim=256, **kwargs):
        self.save_hyperparameters()
        hg = get_hourglass[model_type]
        super().__init__(
            nstack=hg['nstack'],
            channels=hg['channels'],
            modules=hg['modules'],
            num_classes=num_classes,
            cnv_dim=cnv_dim,
            **kwargs
        )

def test_outputs():
    with torch.no_grad():
        hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = model(x)[-1]
        print("hmap_tl", hmap_tl.shape, hmap_tl.min(), hmap_tl.max())
        print("hmap_br", hmap_br.shape, hmap_br.min(), hmap_br.max())
        print("embd_tl", embd_tl.shape)
        print("embd_br", embd_br.shape)
        print("regs_tl", regs_tl.shape)
        print("regs_br", regs_br.shape)

if __name__ == '__main__':
    # from torchsummary import summary
    from torchinfo import summary

    # x = torch.randn(1, 3, 512, 512)
    x = torch.randn(1, 3, 256, 256)
    model = CornerNetPL(model_type='small_hourglass', num_classes=20)
    summary(model, x.shape)
