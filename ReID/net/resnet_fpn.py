# --------------------------------------------------------
# Pytorch Faster R-CNN and FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen, Yixiao Ge
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
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
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
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

class BuildBlock(nn.Module):
  def __init__(self, planes=256):
    super(BuildBlock, self).__init__()
    # Top-down layers, use nn.ConvTranspose2d to replace nn.Conv2d+F.upsample?
    self.toplayer1 = nn.Conv2d(2048, planes, kernel_size=1, stride=1, padding=0)  # Reduce channels
    self.toplayer2 = nn.Conv2d( 256, planes, kernel_size=3, stride=1, padding=1)
    self.toplayer3 = nn.Conv2d( 256, planes, kernel_size=3, stride=1, padding=1)
    self.toplayer4 = nn.Conv2d( 256, planes, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    self.latlayer1 = nn.Conv2d(1024, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer2 = nn.Conv2d( 512, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer3 = nn.Conv2d( 256, planes, kernel_size=1, stride=1, padding=0)

    #self.subsample = nn.AvgPool2d(2, stride=2)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    return F.upsample(x, size=(H,W), mode='bilinear') + y

  def forward(self, c2, c3, c4, c5):
    # Top-down
    p5 = self.toplayer1(c5)
    #p6 = self.subsample(p5)
    p4 = self._upsample_add(p5, self.latlayer1(c4))
    p4 = self.toplayer2(p4)
    p3 = self._upsample_add(p4, self.latlayer2(c3))
    p3 = self.toplayer3(p3)
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    p2 = self.toplayer4(p2)
    
    return p2, p3, p4, p5 #, p6

class HiddenBlock(nn.Module):
  def __init__(self, channels, planes):
    super(HiddenBlock, self).__init__()
    self.fc1 = nn.Linear(channels * 7 * 7,planes)
    self.fc2 = nn.Linear(planes,planes)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    # maxpool different from pytorch-resnet, to match tf-faster-rcnn
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    weights = model_zoo.load_url(model_urls['resnet50'])
    load_dict = dict()
    for k, v in weights.items():
      if 'fc' not in k.split('.'):
        load_dict[k] = v
    model.load_state_dict(load_dict)
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model


class ResNetFPN(nn.Module):
  def __init__(self, num_layers=50, pretrained=True, neck=1, red=1, combine=False, \
                conv_combi=False, squeeze_ext=False, layer_norm=False, \
                  combi_big=False, attention=0, last_only=False, make_small=False):
    super(ResNetFPN, self).__init__()
    self._layers = {}
    self.neck = neck
    self.red = red
    self._feat_stride = [4, 8, 16, 32, 64]
    self._net_conv_channels = 256
    self._fc7_channels = 1024
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self.combine = combine
    self.conv_combi = conv_combi
    self.squeeze_ext = squeeze_ext 
    self.layer_norm = layer_norm
    self.combi_big = combi_big
    self.last_only = last_only
    self.attention = attention
    self.make_small = make_small
    print('Using Combi layer: {}'.format(combine))
    print('Using Conv combi: {}'.format(conv_combi))
    print('Squeeze Ext: {}'.format(squeeze_ext))
    print('Layer norm: {}'.format(layer_norm))
    print('Combi Big: {}'.format(combi_big))
    print('Last only: {}'.format(last_only))
    print('Make small: {}'.format(make_small))

    self._init_head_tail(pretrained=pretrained, red=red, num_layers=num_layers)

  def pool_and_flat(self, feats: torch.tensor):
    return torch.flatten(self.maxpool(feats), 1)

  def forward(self, x, output_option='plain', val=False):
    c2 = self._layers['head'][0](x)
    c3 = self._layers['head'][1](c2)
    c4 = self._layers['head'][2](c3)
    c5 = self._layers['head'][3](c4)
    p2, p3, p4, p5 = self._layers['fpn'](c2, c3, c4, c5)

    if self.attention or self.conv_combi:
      p5_big = self.upsampling(p5) # times 2: 8x4 -> 16x8
      p5_big = self.upsampling(p5_big) # times 2 16x8 -> 32x16
      p5_big = self.upsampling(p5_big) # times 2 32x16 -> 64x32

      p4_big = self.upsampling(p4) # times 2: 16x8 -> 32x16
      p4_big = self.upsampling(p4_big) # times 2: 32x16 -> 64x32

      p3_big = self.upsampling(p3) # times 2: 32x16 -> 64x32

      attention_features = torch.cat([p2, p3_big, p4_big, p5_big], dim=1)
      attention_features = self.conv1x1(attention_features)

    if not self.conv_combi:
      net_conv = [p2, p3, p4, p5]
      p2 = self.pool_and_flat(p2)
      p3 = self.pool_and_flat(p3)
      p4 = self.pool_and_flat(p4)
      p5 = self.pool_and_flat(p5)
      
      if self.red:
        p2 = self.reds[0](p2)
        p3 = self.reds[1](p3)
        p4 = self.reds[2](p4)
        p5 = self.reds[3](p5)

      if self.combine:
        ps = torch.cat([p2, p3, p4, p5], dim=1)
        ps = self.combi_layer(ps)
      elif self.last_only:
        if self.combi_big or self.make_small:
          p5 = self.last_linear(p5)
        ps = p5

      if self.neck and not self.combine and not self.last_only:
        fp2 = self.necks[0](p2)  
        fp3 = self.necks[1](p3)
        fp4 = self.necks[2](p4)
        fp5 = self.necks[3](p5)
      elif self.neck:
        fps = self.necks(ps)
      elif self.combine or self.last_only:
        fps = ps
      else:
          fp2, fp3, fp4, fp5 = p2, p3, p4, p5

      if not self.combine and not self.last_only:
        x2 = self.fcs[0](fp2)
        x3 = self.fcs[1](fp3)
        x4 = self.fcs[2](fp4)
        x5 = self.fcs[3](fp5)

        xs = [x2, x3, x4, x5]
        ps = [p2, p3, p4, p5]
        fps = [fp2, fp3, fp4, fp5]
      else:
        xs = self.fcs(fps)
    
    else:
      ps = self.pool_and_flat(attention_features)

      if self.red:
        ps = self.reds[0](ps)

      if self.neck:
        fps = self.necks(ps)
      else:
        fps = ps

      xs = self.fcs(fps)

    if output_option == 'norm':
      if self.attention:
        return xs, ps, attention_features
      return xs, ps
    elif output_option == 'plain':
      if not self.combine and not self.conv_combi and not self.last_only:
        return xs, [F.normalize(p, p=2, dim=1) for p in ps]
      else:
        return xs, F.normalize(ps, p=2, dim=1)
    elif output_option == 'neck' and self.neck:
        return xs, fps
    elif output_option == 'neck' and not self.neck:
        print("Output option neck only avaiable if bottleneck (neck) is "
              "enabeled - giving back x and fc7")
        return xs, ps

  def _head_to_tail(self, pool5):
    if cfg.FPN:
        fc7 = self.head(pool5)
    else:
        fc7 = self.resnet.layer4(pool5).mean(3).mean(2) # average pooling after layer4
    return fc7

  def _init_head_tail(self, pretrained=True, red=1, num_layers=50, num_classes=1000):
    # choose different blocks for different number of layers
    if num_layers == 50:
        self.resnet = resnet50(pretrained=True)

    elif num_layers == 101:
        self.resnet = resnet101(pretrained=True)

    elif num_layers == 152:
        self.resnet = resnet152(pretrained=True)

    else:
        # other numbers are not supported
        raise NotImplementedError
    '''
    # Fix blocks 
    for p in self.resnet.bn1.parameters(): p.requires_grad=False
    for p in self.resnet.conv1.parameters(): p.requires_grad=False
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
        for p in self.resnet.layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
        for p in self.resnet.layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
        for p in self.resnet.layer1.parameters(): p.requires_grad=False
    
    
    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.resnet.apply(set_bn_fix)
    '''

    # Build resnet.
    # Build Building Block for FPN
    self.fpn_block = BuildBlock()
    self._layers['fpn'] = self.fpn_block
    self._layers['head'] = []
    self._layers['head'].append(nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, 
    self.resnet.maxpool,self.resnet.layer1))
    self._layers['head'].append(nn.Sequential(self.resnet.layer2))
    self._layers['head'].append(nn.Sequential(self.resnet.layer3))
    self._layers['head'].append(nn.Sequential(self.resnet.layer4))
    self.head = HiddenBlock(self._net_conv_channels,self._fc7_channels)
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    if self.attention or self.conv_combi:
      self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')
      if self.combi_big:
        self.feature_map_dim = 2048
      else:
        self.feature_map_dim = 1024
      self.conv1x1 = nn.Conv2d(in_channels=1024, out_channels=self.feature_map_dim, kernel_size=1)

    if red == 1:
      self.red = None
    else:
      reds = []
      for _ in range(4):
        reds.append(nn.Linear(256, int(256/red)))
      self.reds = nn.Sequential(*reds)
      print("reduce output dimension resnet by {}".format(red))

    if self.combine or self.conv_combi or self.last_only:
      if self.combine:
        if self.squeeze_ext and self.layer_norm:
            if self.combi_big:
              self.combi_layer = nn.Sequential(
                      nn.Linear(int(4 * 256/red), int(2 * 256/red)),
                      nn.Linear(int(2 * 256/red), 2048),
                      nn.LayerNorm(2048))
            else:
              self.combi_layer = nn.Sequential(
                    nn.Linear(int(4 * 256/red), int(2 * 256/red)),
                    nn.Linear(int(2 * 256/red), int(4 * 256/red)),
                    nn.LayerNorm(int(4 * 256/red)))
        elif self.squeeze_ext:
          if self.combi_big:
            self.combi_layer = nn.Sequential(
                    nn.Linear(int(4 * 256/red), int(2 * 256/red)),
                    nn.Linear(int(2 * 256/red), 2048))
          else:
            self.combi_layer = nn.Sequential(
                    nn.Linear(int(4 * 256/red), int(2 * 256/red)),
                    nn.Linear(int(2 * 256/red), int(4 * 256/red)))
        else:
            if self.combi_big:
              self.combi_layer = nn.Linear(int(4 * 256/red), 2048)
            else:
              self.combi_layer = nn.Linear(int(4 * 256/red), int(4 * 256/red))
      
      if self.combi_big and self.last_only:
        self.last_linear = nn.Linear(int(256/red), 2048)
      if self.make_small and self.last_only:
        self.last_linear = nn.Linear(int(256/red), 128)

      if self.combi_big:
        dim = 2048
      elif self.last_only:
        if self.make_small:
          dim = 128
        else:
          dim = 256
      else:
        dim = int(4 * 256/red)
      
      if self.neck:
        bottleneck = nn.BatchNorm1d(dim)
        bottleneck.bias.requires_grad_(False)  # no shift
        fc = nn.Linear(dim, num_classes,
                            bias=False)

        bottleneck.apply(weights_init_kaiming)
        fc.apply(weights_init_classifier)
        self.necks = bottleneck
        self.fcs = fc
      else:
        fc = nn.Linear(dim, num_classes)
        self.fcs = fc
    else: # no internal combination
      fcs = []
      for _ in range(4):
        if self.neck:
          neck = []
          bottleneck = nn.BatchNorm1d(int(256/red))
          bottleneck.bias.requires_grad_(False)  # no shift
          fc = nn.Linear(int(256/red), num_classes,
                              bias=False)

          bottleneck.apply(weights_init_kaiming)
          fc.apply(weights_init_classifier)
          neck.append(bottleneck)
          fcs.append(fc)
        else:
          fc = nn.Linear(int(256/red), num_classes)
          fcs.append(fc)
      if self.neck:
        self.necks = nn.Sequential(*neck)
      self.fcs = nn.Sequential(*fcs)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.normal_(m.weight, std=0.001)
    if m.bias:
      nn.init.constant_(m.bias, 0.0)

