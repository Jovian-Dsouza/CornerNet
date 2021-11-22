import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
# from lib.cpool import TopPool, BottomPool, LeftPool, RightPool
from cornerpool import TopPool, BottomPool, LeftPool, RightPool


BN_MOMENTUM = 0.1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


class pool(nn.Module):
  def __init__(self, dim, pool1, pool2):
    super(pool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)

    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)

    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)

    self.conv2 = convolution(3, dim, dim)

    self.pool1 = pool1()
    self.pool2 = pool2()

  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))

    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
    bn1 = self.bn1(self.conv1(x))

    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out


class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()

    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class CornerNet(pl.LightningModule):
  def __init__(self, block, layers, num_layers, head_conv, num_classes):
    super(CornerNet, self).__init__()
    self.inplanes = 64
    self.deconv_with_bias = False
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    # used for deconv layers
    self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])

    self.cnvs_tl = pool(256, TopPool, LeftPool)
    self.cnvs_br = pool(256, BottomPool, RightPool)

    # heatmap layers
    self.hmap_tl = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, num_classes, kernel_size=1))
    self.hmap_br = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, num_classes, kernel_size=1))

    # embedding layers
    self.embd_tl = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 1, kernel_size=1))
    self.embd_br = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 1, kernel_size=1))

    # regression layers
    self.regs_tl = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 2, kernel_size=1))
    self.regs_br = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 2, kernel_size=1))
    self.init_weights(num_layers)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                           kernel_size=1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def _get_deconv_cfg(self, deconv_kernel, index):
    if deconv_kernel == 4:
      padding = 1
      output_padding = 0
    elif deconv_kernel == 3:
      padding = 1
      output_padding = 1
    elif deconv_kernel == 2:
      padding = 0
      output_padding = 0

    return deconv_kernel, padding, output_padding

  def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
    assert num_layers == len(num_filters), \
      'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    assert num_layers == len(num_kernels), \
      'ERROR: num_deconv_layers is different len(num_deconv_filters)'

    layers = []
    for i in range(num_layers):
      kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

      planes = num_filters[i]
      layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                                       out_channels=planes,
                                       kernel_size=kernel,
                                       stride=2,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=self.deconv_with_bias))
      layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
      layers.append(nn.ReLU(inplace=True))
      self.inplanes = planes

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.deconv_layers(x)
    x_tl = self.cnvs_tl(x)
    x_br = self.cnvs_br(x)
    out = [[self.hmap_tl(x_tl), self.hmap_br(x_br),
            self.embd_tl(x_tl), self.embd_br(x_br),
            self.regs_tl(x_tl), self.regs_br(x_br)]]
    return out

  def init_weights(self, num_layers):
    for m in self.deconv_layers.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    for m in self.hmap_tl.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.bias, -2.19)
    for m in self.hmap_br.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.bias, -2.19)
    for m in self.regs_tl.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    for m in self.regs_br.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    for m in self.embd_tl.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    for m in self.embd_br.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    url = model_urls['resnet{}'.format(num_layers)]
    pretrained_state_dict = model_zoo.load_url(url)
    print('=> loading pretrained model {}'.format(url))
    self.load_state_dict(pretrained_state_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

class CornerNetPL(CornerNet):
    def __init__(self, model_type, num_classes, **kwargs):
        self.save_hyperparameters()
        arch, num_layers = model_type.split('_')
        num_layers = int(num_layers)
        block_class, layers = resnet_spec[num_layers]
        super().__init__(
            block_class, 
            layers, 
            num_layers=num_layers, 
            head_conv=64, 
            num_classes=num_classes
        )

if __name__ == '__main__':
  from torchinfo import summary

  x = torch.randn(1, 3, 256, 256)
  model = CornerNetPL('resnet_18', num_classes=20)
  summary(model, x.shape)
