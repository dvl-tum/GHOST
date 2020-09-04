import torch.nn as nn
import math
import pdb, time, sys
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet200', 'resnet1001']

def _resnet(arch, block, layers, pretrained, progress, last_stride=0, neck=0, **kwargs):
    print(block)
    model = ResNetCheck(block, layers, last_stride=last_stride, neck=neck, **kwargs)
    if pretrained:
        if not neck:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
            model.load_state_dict(state_dict)
        else:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

            state_dict_new = dict()
            for k, v in state_dict.items():
                if k.split('.')[0][:-1] != 'layer':
                    state_dict_new['features.' + k] = v
                else:
                    state_dict_new['features.' + 'Bottleneck_' + k.split('.')[0][-1] + '_' + k.split('.')[1] + '.' +('.').join(k.split('.')[2:])] = v

            #state_dict = {'features.' + 'Bottleneck_' + k.split('.')[0][-1] + '_' + k.split('.')[1] + '.' +('.').join(k.split('.')[2:]): v for k, v in state_dict.items()} 
            for i in state_dict_new:
                if 'fc' in i:
                    continue
                model.state_dict()[i].copy_(state_dict_new[i])
    return model

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(pretrained=False, progress=True, last_stride=0, neck=0, **kwargs):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    last_stride, neck, **kwargs)
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

# looks like the ResNet1001 was only defined for the cifar dataset
def resnet1001(pretrained=False, **kwargs):
    # the input data should be like 32 x 3 x 32 x 32
    model = PreActResNet(PreActBottleneck, [111, 111, 111], **kwargs)
    return model

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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
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

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetCheck(nn.Module):

    def __init__(self, block, layers, num_classes=10, neck=0, last_stride=0):
        print(block)
        self.inplanes = 64
        self.dilation = 1

        self.neck = neck
        self.last_stride = last_stride
        super(ResNetCheck, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(self.inplanes)),
            ('relu', nn.ReLU(inplace=True)),
            ('max_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self._make_layer(block, 64, layers[0], stage=1)
        self._make_layer(block, 128, layers[1], stride=2, stage=2)
        self._make_layer(block, 256, layers[2], stride=2, stage=3)
        self._make_layer(block, 512, layers[3], stride=2, stage=4)
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        
        # student
        '''
        d_embed = 512 * block.expansion
        d_hid = 4 * d_embed
        dropout = 0.4
        self.linear1 = nn.Linear(d_embed, d_hid)
        self.dropout = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(d_hid, d_hid)
        self.linear2 = nn.Linear(d_hid, d_embed)
        self.activation = F.relu 
        '''
        if self.neck:
            self.bottleneck = nn.BatchNorm1d(512 * block.expansion)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc = nn.Linear(512 * block.expansion, num_classes,
                                        bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, stage=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #downsample = nn.Sequential(
            #    nn.BatchNorm2d(self.inplanes),
            #    nn.Conv2d(self.inplanes, planes * block.expansion,
            #              kernel_size=1, stride=stride, bias=False),
            #)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.features.add_module(
            'Bottleneck_%d_%d' % (stage, 0),
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.features.add_module(
                'Bottleneck_%d_%d' % (stage, i),
                block(self.inplanes, planes))

    def forward(self, x, output_option='norm', chunks=3):
        modules = [module for k, module in self._modules.items()][0]
        x = checkpoint_sequential(modules, chunks, x.requires_grad_())
        print(x.requires_grad)
        fc7 = x.view(x.size(0), -1)

        # here student can be added
        feats = fc7
        #feats = self.linear2(self.dropout(self.activation(self.linear1(fc7)))) 

        if self.neck:
            feats_after = self.bottleneck(feats)
        else:
            feats_after = feats

        x = self.fc(feats_after)

        if output_option == 'norm':
            return x, fc7, feats
        elif output_option == 'plain':
            return x, F.normalize(fc7, p=2, dim=1), F.normalize(feats, p=2,
                                                                dim=1)
        elif output_option == 'neck' and self.neck:
            return x, fc7, feats_after
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, fc7, feats



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
